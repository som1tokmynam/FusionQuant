import os
import subprocess
import shutil
import tempfile
import threading
from huggingface_hub import HfApi, ModelCard, ModelCardData, whoami, snapshot_download
from pathlib import Path
from textwrap import dedent
from typing import List, Tuple, Optional, Callable, Dict, Union, Any

# --- Path Definitions ---
APP_SCRIPT_WORKDIR = Path(os.environ.get("APP_DIR", "/home/builder/app"))
LLAMA_CPP_SOURCE_DIR = Path(os.environ.get("LLAMA_CPP_DIR", APP_SCRIPT_WORKDIR / "llama.cpp"))

# --- Default Logger for Path Setup ---
def _default_log_fn_for_path_setup(message: str, level: str = "INFO"):
    print(f"[{level}] (gguf_utils_path_setup): {message}")

# --- Executable Finder ---
def find_executable(
    exe_name: str,
    log_fn: Callable[[str, str], None] = _default_log_fn_for_path_setup
) -> str:
    log_fn(f"Searching for executable '{exe_name}'...", "DEBUG")
    resolved_path_from_which = shutil.which(exe_name)
    if resolved_path_from_which:
        log_fn(f"Found executable '{exe_name}' in system PATH at: {resolved_path_from_which}", "INFO")
        return resolved_path_from_which
    
    potential_path_in_llama_build = LLAMA_CPP_SOURCE_DIR / "build" / "bin" / exe_name
    if potential_path_in_llama_build.is_file() and os.access(potential_path_in_llama_build, os.X_OK):
        log_fn(f"Found executable '{exe_name}' in llama.cpp build dir: {potential_path_in_llama_build}", "INFO")
        return str(potential_path_in_llama_build)
        
    potential_path_in_script_workdir = APP_SCRIPT_WORKDIR / exe_name
    if potential_path_in_script_workdir.is_file() and os.access(potential_path_in_script_workdir, os.X_OK):
        log_fn(f"Found executable '{exe_name}' in APP_SCRIPT_WORKDIR: {potential_path_in_script_workdir}", "INFO")
        return str(potential_path_in_script_workdir)

    error_msg = (
        f"CRITICAL: Executable '{exe_name}' not found in system PATH or common application directories. "
        f"Ensure '{exe_name}' is compiled and its location is in the PATH environment variable "
        f"inside the Docker container. Searched PATH and common locations like {LLAMA_CPP_SOURCE_DIR}/build/bin."
    )
    log_fn(error_msg, "ERROR")
    raise FileNotFoundError(error_msg)

# --- Define Executable Paths ---
LLAMA_QUANTIZE_PATH = find_executable("llama-quantize")
LLAMA_CLI_PATH = find_executable("llama-cli") 
try:
    LLAMA_IMATRIX_PATH = find_executable("llama-imatrix")
except FileNotFoundError:
    _default_log_fn_for_path_setup("WARNING: Executable 'llama-imatrix' not found. Importance matrix functionality will be impaired.", "WARNING")
    LLAMA_IMATRIX_PATH = None 
try:
    LLAMA_GGUF_SPLIT_PATH = find_executable("llama-gguf-split")
except FileNotFoundError:
    _default_log_fn_for_path_setup("WARNING: Executable 'llama-gguf-split' not found. GGUF splitting functionality will be impaired.", "WARNING")
    LLAMA_GGUF_SPLIT_PATH = None

CONVERSION_SCRIPT = LLAMA_CPP_SOURCE_DIR / "convert_hf_to_gguf.py"
if not (CONVERSION_SCRIPT.is_file() and os.access(CONVERSION_SCRIPT, os.R_OK)):
    err_msg_conv = f"CRITICAL: Python script 'convert_hf_to_gguf.py' not found or not readable in {LLAMA_CPP_SOURCE_DIR}"
    _default_log_fn_for_path_setup(err_msg_conv, "ERROR")
    CONVERSION_SCRIPT = Path("PYTHON_CONVERSION_SCRIPT_NOT_FOUND.py") # Placeholder


# --- Custom Exception ---
class GGUFConversionError(Exception):
    """Custom exception for GGUF conversion errors."""
    pass

# --- Helper: Run GGUF Command with Logging ---
def _run_gguf_command_with_logging(
    cmd_list: List[str],
    log_fn: Callable[[str, str], None],
    cwd: Optional[Union[str, Path]] = None,
    env: Optional[Dict[str, str]] = None,
    timeout: Optional[int] = 7200,
    operation_name: str = "GGUF operation"
):
    log_fn(f"Running {operation_name}: {' '.join(cmd_list)}", "INFO")
    if cwd: log_fn(f"Working directory: {cwd}", "DEBUG")

    effective_env = os.environ.copy()
    if env: effective_env.update(env)
    effective_env["PYTHONUNBUFFERED"] = "1"

    process = subprocess.Popen(
        cmd_list, cwd=str(cwd) if cwd else None, env=effective_env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, universal_newlines=True, errors='replace'
    )

    def read_stream():
        try:
            for line in iter(process.stdout.readline, ''): 
                if line: log_fn(line.rstrip('\n\r'), "INFO")
        except Exception as e_read:
            log_fn(f"Error reading stream for {operation_name}: {e_read}", "ERROR")
        finally:
            if process.stdout: process.stdout.close() 

    thread = threading.Thread(target=read_stream)
    thread.daemon = True
    thread.start()

    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        if timeout is not None:
            log_fn(f"{operation_name} timed out after {timeout} seconds. Terminating...", "ERROR")
            process.terminate()
            try: process.wait(timeout=30)
            except subprocess.TimeoutExpired: process.kill(); process.wait()
            thread.join(timeout=10)
            raise GGUFConversionError(f"{operation_name} timed out: {' '.join(cmd_list)}")

    thread.join(timeout=30)
    if process.returncode != 0:
        raise GGUFConversionError(f"{operation_name} failed (exit code {process.returncode}): {' '.join(cmd_list)}")
    log_fn(f"{operation_name} completed successfully.", "INFO")

# --- Importance Matrix Generation ---
def generate_importance_matrix(
    model_path: str, 
    train_data_path: str, 
    output_imatrix_path: str, 
    log_fn: Callable[[str, str], None],
    threads: int = 4 
) -> bool:
    log_fn(f"Attempting to generate importance matrix for '{model_path}' using '{train_data_path}'", "INFO")
    if not LLAMA_IMATRIX_PATH:
        log_fn("'llama-imatrix' executable not found. Cannot generate importance matrix.", "ERROR")
        return False
    if not Path(train_data_path).exists():
        log_fn(f"Training data for importance matrix not found: '{train_data_path}'", "ERROR")
        return False
    
    imatrix_cmd = [
        LLAMA_IMATRIX_PATH,
        "-m", model_path,
        "-f", train_data_path,
        "--verbosity", "1", 
        "-o", output_imatrix_path,
        "-t", str(threads) 
    ]
    try:
        _run_gguf_command_with_logging(
            imatrix_cmd, 
            log_fn, 
            operation_name="Importance Matrix Generation",
            timeout=None
        )
        if Path(output_imatrix_path).exists():
            log_fn(f"Importance matrix generated successfully: {output_imatrix_path}", "INFO")
            return True
        else:
            log_fn("Importance matrix generation command ran, but output file was not found.", "ERROR")
            return False
    except GGUFConversionError as e:
        log_fn(f"Failed to generate importance matrix: {e}", "ERROR")
        return False
    except Exception as e_unhandled:
        log_fn(f"Unexpected error during importance matrix generation: {e_unhandled}", "CRITICAL_ERROR")
        return False

# --- Main Processing Function for GGUF Conversion ---
def process_gguf_conversion(
    model_source_type: str,
    log_fn: Callable[[str, str], None],
    result_container: Dict[str, Any],
    output_dir_gguf: str,
    temp_dir_base_for_job: str,
    download_dir_for_job: str,
    hf_token: Optional[str] = None,
    hf_model_id: Optional[str] = None,
    local_model_path: Optional[str] = None,
    custom_model_name_gguf: Optional[str] = None,
    quant_methods_list: List[str] = ["Q4_K_M"],
    use_imatrix_bool: bool = False,
    imatrix_quant_methods_list: Optional[List[str]] = None,
    upload_to_hf_bool: bool = False,
    hf_repo_private_bool: bool = False,
    train_data_path_gguf: Optional[str] = None,
    user_specified_gguf_save_path: bool = True,
    split_model_bool: bool = False,
    split_max_tensors_val: int = 256,
    split_max_size_val: Optional[str] = None
):
    result_container['final_status_messages_list'] = []
    result_container['error_message'] = None

    source_model_dir_for_conversion: Path
    model_name_base: str

    try:
        log_fn("=== Starting GGUF Conversion Process ===", "INFO")
        Path(temp_dir_base_for_job).mkdir(parents=True, exist_ok=True)
        log_fn(f"Ensured temporary job base directory exists: {temp_dir_base_for_job}", "DEBUG")

        if model_source_type == "HF Hub":
            if not hf_model_id:
                raise GGUFConversionError("Hugging Face Model ID is required for HF Hub source.")
            log_fn(f"Using HF Hub model: {hf_model_id}", "INFO")
            model_specific_download_path = Path(download_dir_for_job) / Path(hf_model_id).name
            model_specific_download_path.mkdir(parents=True, exist_ok=True)
            log_fn(f"Downloading {hf_model_id} to {model_specific_download_path}...", "INFO")
            try:
                actual_downloaded_path_str = snapshot_download(
                    repo_id=hf_model_id,
                    local_dir=str(model_specific_download_path),
                    local_dir_use_symlinks=False,
                    token=hf_token,
                )
                source_model_dir_for_conversion = Path(actual_downloaded_path_str)
                log_fn(f"Model {hf_model_id} downloaded successfully to {source_model_dir_for_conversion}.", "INFO")
            except Exception as e_dl:
                raise GGUFConversionError(f"Failed to download model {hf_model_id}: {e_dl}")
            model_name_base = custom_model_name_gguf or Path(hf_model_id).name
        elif model_source_type == "Local Path":
            if not local_model_path or not Path(local_model_path).is_dir():
                raise GGUFConversionError("Valid local model path is required for Local Path source.")
            source_model_dir_for_conversion = Path(local_model_path)
            log_fn(f"Using local model from: {source_model_dir_for_conversion}", "INFO")
            model_name_base = custom_model_name_gguf or source_model_dir_for_conversion.name
        else:
            raise GGUFConversionError(f"Invalid model_source_type: {model_source_type}")

        Path(output_dir_gguf).mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="gguf_BF16conv_", dir=temp_dir_base_for_job) as Bf16_conversion_workdir_str:
            Bf16_conversion_workdir = Path(Bf16_conversion_workdir_str)
            initial_gguf_filename = f"{model_name_base}.BF16.gguf"
            initial_gguf_path = Bf16_conversion_workdir / initial_gguf_filename

            convert_cmd = [
                "python3", str(CONVERSION_SCRIPT), str(source_model_dir_for_conversion),
                "--outfile", str(initial_gguf_path),
                "--outtype", "bf16"
            ]
            log_fn(f"Converting model to initial GGUF (BF16)... Output: {initial_gguf_path}", "INFO")
            _run_gguf_command_with_logging(convert_cmd, log_fn, operation_name="HF to Initial GGUF Conversion")
            if not initial_gguf_path.exists():
                raise GGUFConversionError(f"Initial GGUF conversion failed: {initial_gguf_path} not found.")

            imatrix_data_path_str: Optional[str] = None
            if use_imatrix_bool:
                if not train_data_path_gguf or not Path(train_data_path_gguf).exists():
                    log_fn("Importance matrix requested, but valid training data path not provided. Skipping imatrix.", "WARNING")
                elif not LLAMA_IMATRIX_PATH:
                    log_fn("Importance matrix requested, but llama-imatrix executable not found. Skipping imatrix.", "WARNING")
                else:
                    imatrix_output_filename = f"{model_name_base}.imatrix.dat"
                    imatrix_data_path_obj = Bf16_conversion_workdir / imatrix_output_filename
                    if generate_importance_matrix(str(initial_gguf_path), train_data_path_gguf, str(imatrix_data_path_obj), log_fn):
                        imatrix_data_path_str = str(imatrix_data_path_obj)
                    else:
                        log_fn("Importance matrix generation failed. Proceeding without it.", "WARNING")
            
            if use_imatrix_bool:
                log_fn(f"IMatrix mode potentially enabled. Using quantization methods: {imatrix_quant_methods_list}", "INFO")
                if imatrix_data_path_str and imatrix_quant_methods_list:
                     all_quant_methods = list(imatrix_quant_methods_list)
                else:
                    all_quant_methods = list(quant_methods_list) if quant_methods_list else []
                    if imatrix_data_path_str:
                        log_fn("IMatrix generated, but no specific imatrix_quant_methods_list provided. Will use standard quant_methods_list if imatrix is applicable by those methods.", "INFO")
                    else:
                        log_fn("IMatrix not used or generation failed. Using standard quantization methods.", "INFO")
            else:
                log_fn(f"Standard quantization enabled. Using methods: {quant_methods_list}", "INFO")
                all_quant_methods = list(quant_methods_list) if quant_methods_list else []

            if not all_quant_methods:
                log_fn("No quantization methods selected. Copying initial BF16 GGUF to output.", "INFO")
                if initial_gguf_path.exists():
                    final_f16_path = Path(output_dir_gguf) / initial_gguf_filename
                    shutil.copy2(initial_gguf_path, final_f16_path)
                    msg = f"✅ Initial GGUF (BF16) created. | Local: `{str(final_f16_path.resolve())}`"
                    result_container['final_status_messages_list'].append(msg)

            for method in all_quant_methods:
                log_fn(f"--- Processing GGUF quantization method: {method} ---", "INFO")
                quant_output_filename = f"{model_name_base}-{method}.gguf"
                quant_output_path_final = Path(output_dir_gguf) / quant_output_filename

                quant_cmd = [LLAMA_QUANTIZE_PATH]
                if imatrix_data_path_str and use_imatrix_bool:
                    is_method_for_imatrix = True
                    if imatrix_quant_methods_list:
                        is_method_for_imatrix = method in imatrix_quant_methods_list
                    
                    if is_method_for_imatrix:
                        log_fn(f"Using importance matrix for {method}: {imatrix_data_path_str}", "INFO")
                        quant_cmd.extend(["--imatrix", imatrix_data_path_str])
                    else:
                        log_fn(f"IMatrix available, but method {method} not designated for imatrix use. Quantizing without imatrix.", "INFO")
                
                quant_cmd.extend([str(initial_gguf_path), str(quant_output_path_final), method])

                log_fn(f"Running quantization for {method}... Output: {quant_output_path_final}", "INFO")
                _run_gguf_command_with_logging(quant_cmd, log_fn, operation_name=f"GGUF Quantization ({method})")

                if not quant_output_path_final.exists():
                    raise GGUFConversionError(f"Quantization for method {method} failed: output file '{quant_output_path_final}' not found.")

                current_quant_status_msg = f"✅ **{method}:** Quantization successful."
                
                is_sharded = False
                if split_model_bool:
                    if not LLAMA_GGUF_SPLIT_PATH:
                        log_fn("Sharding requested, but 'llama-gguf-split' executable not found. Skipping.", "WARNING")
                        current_quant_status_msg += " | Sharding Skipped (executable not found)"
                    else:
                        log_fn(f"Attempting to shard {quant_output_filename}...", "INFO")
                        split_cmd = [LLAMA_GGUF_SPLIT_PATH]
                        if split_max_size_val and split_max_size_val.strip():
                            split_cmd.extend(["--split-max-size", split_max_size_val])
                        if split_max_tensors_val is not None:
                             split_cmd.extend(["--split-max-tensors", str(split_max_tensors_val)])
                        split_cmd.append(str(quant_output_path_final))
                        
                        try:
                            _run_gguf_command_with_logging(split_cmd, log_fn, operation_name=f"GGUF Sharding ({method})")
                            is_sharded = True
                            log_fn(f"Sharding successful for {method} quant.", "INFO")
                            current_quant_status_msg += " | Sharding Successful"
                        except GGUFConversionError as e_split:
                            log_fn(f"Sharding failed for {method} quant: {e_split}", "ERROR")
                            current_quant_status_msg += f" | Sharding Failed: {e_split}"

                local_quant_file_kept_and_exists = True

                if upload_to_hf_bool:
                    if not hf_token:
                        log_fn(f"Cannot upload {method} GGUF: Hugging Face token not provided.", "ERROR")
                        current_quant_status_msg += " | HF Upload Failed (token missing)"
                    else:
                        hf_username = whoami(token=hf_token).get('name')
                        base_repo_name_for_gguf = custom_model_name_gguf or model_name_base
                        gguf_repo_id = f"{hf_username}/{base_repo_name_for_gguf}-GGUF" if '/' not in base_repo_name_for_gguf else f"{base_repo_name_for_gguf}-GGUF"
                        
                        try:
                            log_fn(f"Ensuring repo exists and preparing to upload to {gguf_repo_id}", "INFO")
                            api = HfApi(token=hf_token)
                            repo_url_obj = api.create_repo(repo_id=gguf_repo_id, private=hf_repo_private_bool, exist_ok=True, repo_type="model")
                            repo_url = repo_url_obj.repo_url if hasattr(repo_url_obj, 'repo_url') else str(repo_url_obj)

                            upload_path_display = ""
                            if is_sharded:
                                log_fn(f"Uploading sharded GGUF for {method} to {gguf_repo_id}", "INFO")
                                shard_files_to_upload = list(Path(output_dir_gguf).glob(f"{Path(quant_output_filename).stem}-*.gguf"))

                                if not shard_files_to_upload:
                                    raise GGUFConversionError(f"Sharding reported success for {method}, but no shard files found with pattern '{Path(quant_output_filename).stem}-*.gguf' in '{output_dir_gguf}'")

                                for shard_path in shard_files_to_upload:
                                    api.upload_file(
                                        path_or_fileobj=str(shard_path),
                                        path_in_repo=shard_path.name,
                                        repo_id=gguf_repo_id,
                                        commit_message=f"Add {method} GGUF quant shard: {shard_path.name}")
                                upload_path_display = f"{len(shard_files_to_upload)} shards"
                                log_fn(f"Successfully uploaded {len(shard_files_to_upload)} shards to {repo_url}", "INFO")
                                
                                if quant_output_path_final.exists() and any(sf != quant_output_path_final for sf in shard_files_to_upload):
                                    log_fn(f"Original pre-split file {quant_output_path_final} still exists after sharding. Deleting it.", "DEBUG")
                                    quant_output_path_final.unlink(missing_ok=True)

                            else: # Not sharded, upload the single file
                                if not quant_output_path_final.exists():
                                     raise GGUFConversionError(f"Quantized file {quant_output_path_final} expected but not found before upload.")
                                api.upload_file(
                                    path_or_fileobj=str(quant_output_path_final),
                                    path_in_repo=quant_output_filename,
                                    repo_id=gguf_repo_id,
                                    commit_message=f"Add {method} GGUF quant: {quant_output_filename}")
                                upload_path_display = quant_output_filename
                                log_fn(f"Successfully uploaded {quant_output_filename} to {repo_url}", "INFO")
                            
                            # --- CORRECTED MODEL CARD SECTION ---
                            card_model_name_title = gguf_repo_id.split('/')[-1]
                            
                            all_methods_str = ", ".join(f"`{m}`" for m in all_quant_methods)
                            
                            source_model_identifier = "Unknown"
                            if model_source_type == "HF Hub" and hf_model_id:
                                source_model_identifier = f"[{hf_model_id}](https://huggingface.co/{hf_model_id})"
                            elif model_source_type == "Local Path" and local_model_path:
                                source_model_identifier = f"`{Path(local_model_path).name}`"
                            
                            card_body_content = dedent(f"""
                            ---
                            license: mit
                            library_name: llama.cpp
                            tags:
                            - gguf
                            - {method.lower().replace("_", "-")}
                            ---
                            # {card_model_name_title}

                            GGUF model files for `{model_name_base}`.

                            This repository contains GGUF models quantized using [`llama.cpp`](https://github.com/ggerganov/llama.cpp).

                            - **Base Model:** {source_model_identifier}
                            - **Quantization Methods Processed in this Job:** {all_methods_str}
                            - **Importance Matrix Used:** {'Yes' if imatrix_data_path_str else 'No'}

                            This specific upload is for the **`{method}`** quantization.
                            """)

                            ModelCard(card_body_content).push_to_hub(gguf_repo_id, token=hf_token)
                            # --- END OF CORRECTION ---

                            current_quant_status_msg += f" | HF: <a href='{repo_url}' target='_blank'>{gguf_repo_id} ({upload_path_display})</a>"

                            if not user_specified_gguf_save_path:
                                if is_sharded:
                                    for p in shard_files_to_upload: 
                                        if p.exists(): p.unlink(missing_ok=True)
                                    log_fn(f"Deleted local shard files for {method} from default path after HF upload.", "INFO")
                                else:
                                    if quant_output_path_final.exists():
                                        quant_output_path_final.unlink(missing_ok=True)
                                        log_fn(f"Deleted local file {quant_output_path_final} for {method} from default path after HF upload.", "INFO")
                                local_quant_file_kept_and_exists = False

                        except Exception as e_upload_gguf:
                            err_upload_msg = f"Failed to upload to {gguf_repo_id}: {e_upload_gguf}"
                            log_fn(err_upload_msg, "ERROR")
                            current_quant_status_msg += f" | HF Upload Error: {str(e_upload_gguf)[:100]}..."
                            
                            current_error_message = result_container.get('error_message')
                            if current_error_message is None:
                                result_container['error_message'] = err_upload_msg
                            else:
                                result_container['error_message'] = (str(current_error_message) + f"\n{err_upload_msg}").strip()
                
                if local_quant_file_kept_and_exists:
                    if is_sharded:
                        existing_shards = [s for s in Path(output_dir_gguf).glob(f"{Path(quant_output_filename).stem}-*.gguf") if s.exists()]
                        if existing_shards:
                             current_quant_status_msg += f" | Local (sharded): `{str(Path(output_dir_gguf).resolve())}` ({len(existing_shards)} shards)"
                        else:
                             current_quant_status_msg += f" | Local (sharded): All shards appear to have been removed or were not found."
                    else:
                        if quant_output_path_final.exists():
                            current_quant_status_msg += f" | Local: `{str(quant_output_path_final.resolve())}`"
                        else:
                            current_quant_status_msg += f" | Local: File {quant_output_path_final.name} appears to have been removed or was not found."

                elif not user_specified_gguf_save_path:
                    current_quant_status_msg += " | Local copy (default path) deleted after successful HF upload."
                
                result_container['final_status_messages_list'].append(current_quant_status_msg)

        log_fn("Initial GGUF conversion temporary workdir auto-cleaned.", "INFO")

    except GGUFConversionError as e_gguf:
        log_fn(f"GGUF Conversion Error: {e_gguf}", "ERROR")
        result_container['error_message'] = str(e_gguf)
    except Exception as e_main:
        import traceback
        tb_str = traceback.format_exc()
        log_fn(f"Unexpected error in GGUF processing: {e_main}\nTraceback:\n{tb_str}", "CRITICAL_ERROR")
        result_container['error_message'] = f"Unexpected critical error in GGUF processing: {str(e_main)}"