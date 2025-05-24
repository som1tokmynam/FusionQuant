import os
import subprocess
import signal # Not actively used in current subprocess calls, but good for reference
import shutil
import tempfile
import threading 
from huggingface_hub import HfApi, ModelCard, whoami
from pathlib import Path
from textwrap import dedent
from typing import List, Tuple, Optional, Any, Callable, Dict # Added Callable, Dict

# Configuration
CONVERSION_SCRIPT = "./llama.cpp/convert_hf_to_gguf.py"
LLAMA_CPP_BASE_PATH = "./llama.cpp"
LLAMA_CPP_BUILD_PATH = os.path.join(LLAMA_CPP_BASE_PATH, "build")

def find_executable(exe_name: str) -> str:
    possible_paths = [
        os.path.join(LLAMA_CPP_BUILD_PATH, "bin", exe_name),
        os.path.join(LLAMA_CPP_BUILD_PATH, exe_name),
        os.path.join(LLAMA_CPP_BASE_PATH, exe_name),
        exe_name
    ]
    for path in possible_paths:
        resolved_path = shutil.which(path)
        if resolved_path:
            # print(f"Found executable {exe_name} at {resolved_path}")
            return resolved_path
    fallback_path = os.path.join(LLAMA_CPP_BUILD_PATH, "bin", exe_name)
    print(f"Warning: Executable {exe_name} not found in checked paths {possible_paths}, defaulting to {fallback_path}")
    return fallback_path

LLAMA_QUANTIZE_PATH = find_executable("llama-quantize")
LLAMA_IMATRIX_PATH = find_executable("llama-imatrix")
LLAMA_GGUF_SPLIT_PATH = find_executable("llama-gguf-split")

class GGUFConversionError(Exception):
    """Custom exception for GGUF conversion errors."""
    pass

def escape(s: str) -> str:
    s = s.replace("&", "&amp;")
    s = s.replace("<", "&lt;")
    s = s.replace(">", "&gt;")
    s = s.replace('"', "&quot;")
    # Avoid replacing \n with <br/> if logs are meant for plain text streaming displays mostly
    # s = s.replace("\n", "<br/>")
    return s

def _run_gguf_command_with_logging(
    cmd_list: List[str],
    log_fn: Callable,
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    timeout: int = 3600,
    operation_name: str = "GGUF operation"
):
    log_fn(f"Running {operation_name}: {' '.join(cmd_list)}", "INFO")
    if cwd: log_fn(f"Working directory: {cwd}", "DEBUG")
    full_env = os.environ.copy()
    if env: full_env.update(env)

    process = subprocess.Popen(
        cmd_list, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, universal_newlines=True, cwd=cwd, env=full_env, errors='replace'
    )
    def read_stream():
        try:
            for line in iter(process.stdout.readline, ''):
                if line: log_fn(line.rstrip('\n\r')) 
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
        process.kill()
        try: process.wait(timeout=5)
        except: pass
        thread.join(timeout=5)
        error_msg = f"{operation_name} timed out after {timeout} seconds: {' '.join(cmd_list)}"
        log_fn(error_msg, "ERROR")
        raise GGUFConversionError(error_msg)
    thread.join(timeout=10)
    if process.returncode != 0:
        error_msg = f"{operation_name} failed with exit code {process.returncode}: {' '.join(cmd_list)}"
        log_fn(error_msg, "ERROR")
        raise GGUFConversionError(error_msg)
    log_fn(f"{operation_name} completed successfully.", "INFO")

def generate_importance_matrix(
    model_path: str, train_data_path: str, output_path: str, log_fn: Callable, ngl: int = 99
):
    log_fn(f"Verifying llama-imatrix executable: {LLAMA_IMATRIX_PATH}", "DEBUG")
    if not (Path(LLAMA_IMATRIX_PATH).is_file() and os.access(LLAMA_IMATRIX_PATH, os.X_OK)):
        raise GGUFConversionError(f"llama-imatrix executable not found or not executable at {LLAMA_IMATRIX_PATH}")
    imatrix_command = [
        LLAMA_IMATRIX_PATH, "-m", model_path, "-f", train_data_path,
        "-ngl", str(ngl), "--output-frequency", "10", "-o", output_path,
    ]
    _run_gguf_command_with_logging(imatrix_command, log_fn, operation_name="Importance Matrix Generation")

def split_upload_model(
    model_path: str, outdir: str, repo_id: Optional[str], hf_token: Optional[str],
    log_fn: Callable, split_max_tensors: int = 256, split_max_size: Optional[str] = None
) -> Tuple[bool, List[str]]:
    log_fn(f"Verifying llama-gguf-split executable: {LLAMA_GGUF_SPLIT_PATH}", "DEBUG")
    if not (Path(LLAMA_GGUF_SPLIT_PATH).is_file() and os.access(LLAMA_GGUF_SPLIT_PATH, os.X_OK)):
        raise GGUFConversionError(f"llama-gguf-split executable not found or not executable at {LLAMA_GGUF_SPLIT_PATH}")
    split_cmd = [LLAMA_GGUF_SPLIT_PATH, "--split"]
    if split_max_size: split_cmd.extend(["--split-max-size", split_max_size])
    else: split_cmd.extend(["--split-max-tensors", str(split_max_tensors)])
    model_path_obj = Path(model_path)
    shard_prefix_path = str(model_path_obj.with_suffix(''))
    split_cmd.extend([model_path, shard_prefix_path])
    log_fn(f"Splitting model with command: {' '.join(split_cmd)}", "INFO")
    try:
        _run_gguf_command_with_logging(split_cmd, log_fn, cwd=outdir, operation_name="Model Splitting")
    except GGUFConversionError as e:
        log_fn(f"Model splitting failed: {e}", "ERROR")
        raise
    log_fn("Model splitting command finished.", "INFO")
    base_shard_name = Path(shard_prefix_path).name
    sharded_files_basenames = [f for f in os.listdir(outdir) if f.startswith(base_shard_name) and f.endswith(".gguf") and "-of-" in f]
    log_fn(f"Found {len(sharded_files_basenames)} sharded files: {sharded_files_basenames}", "INFO")
    upload_success = True
    if sharded_files_basenames and repo_id and hf_token:
        api = HfApi(token=hf_token)
        log_fn(f"Uploading {len(sharded_files_basenames)} sharded files to {repo_id}...", "INFO")
        try:
            for shard_basename in sharded_files_basenames:
                log_fn(f"Uploading shard: {shard_basename}", "DEBUG")
                api.upload_file(
                    path_or_fileobj=os.path.join(outdir, shard_basename), path_in_repo=shard_basename, repo_id=repo_id
                )
            log_fn(f"Successfully uploaded sharded files to {repo_id}.", "INFO")
        except Exception as e:
            log_fn(f"Upload error for sharded files to {repo_id}: {e}", "ERROR"); upload_success = False
    elif not sharded_files_basenames: log_fn("No sharded files found to upload.", "WARNING")
    return upload_success, sharded_files_basenames

def process_gguf_conversion(
    model_source: str, hf_token_gguf: Optional[str], log_fn: Callable, result_container: Dict,
    model_id: Optional[str] = None, local_model_path: Optional[str] = None,
    custom_output_name: Optional[str] = None, # <-- MODIFIED: Added new parameter
    q_methods: Optional[List[str]] = None, use_imatrix: bool = False,
    imatrix_q_methods: Optional[List[str]] = None, upload_to_hf: bool = False,
    private_repo: bool = False, train_data_file: Optional[Any] = None,
    local_output_path: Optional[str] = None, split_model: bool = False,
    split_max_tensors: int = 256, split_max_size: Optional[str] = None,
    base_outputs_dir: str = "outputs", base_downloads_dir: str = "downloads"
):
    result_container['final_html'], result_container['image_path'], result_container['error_msg'] = "", None, None
    model_name_for_files, new_repo_id_gguf_for_upload = "", ""
    all_saved_files_info_local = []
    try:
        log_fn(f"GGUF Utils CWD: {os.getcwd()}", "DEBUG")

        # --- MODIFIED: Section for model_name_for_files determination ---
        if custom_output_name and custom_output_name.strip():
            model_name_for_files = custom_output_name.strip()
            log_fn(f"Using custom base name for GGUF files: {model_name_for_files}", "INFO")
        elif model_source == "HF Hub":
            if not model_id: raise GGUFConversionError("HF model ID required.")
            model_name_for_files = model_id.split('/')[-1]
        elif model_source == "Local Path":
            if not local_model_path: raise GGUFConversionError("Local model path required.")
            model_name_for_files = Path(local_model_path).name
            if not Path(local_model_path).exists(): raise GGUFConversionError(f"Local model path DNE: {local_model_path}")
        else: raise GGUFConversionError(f"Invalid model_source: {model_source}")

        quant_methods_to_run = imatrix_q_methods if use_imatrix and imatrix_q_methods else q_methods
        quant_methods_to_run = quant_methods_to_run if quant_methods_to_run is not None else []
        if not quant_methods_to_run and not (q_methods and not use_imatrix):
            log_fn("No quantization selected. Only BF16 GGUF if not already present.", "INFO")


        hf_api_for_gguf_upload = None
        if upload_to_hf:
            if not hf_token_gguf: raise GGUFConversionError("HF token required for GGUF upload.")
            try:
                hf_api_for_gguf_upload = HfApi(token=hf_token_gguf)
                user_info = whoami(token=hf_token_gguf)
                username = user_info['name']
                new_repo_id_gguf_for_upload = f"{username}/{model_name_for_files}-GGUF"
                hf_api_for_gguf_upload.create_repo(repo_id=new_repo_id_gguf_for_upload, exist_ok=True, private=private_repo)
                log_fn(f"Ensured HF repo for GGUF: {new_repo_id_gguf_for_upload}", "INFO")
            except Exception as e: raise GGUFConversionError(f"Failed HF auth/repo creation for GGUF: {str(e)}")

        abs_base_outputs_dir = Path(base_outputs_dir).resolve()
        abs_base_outputs_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix=f"{model_name_for_files}_gguf_proc_", dir=abs_base_outputs_dir) as temp_processing_dir_str:
            temp_processing_dir = Path(temp_processing_dir_str)
            log_fn(f"Temp processing dir: {temp_processing_dir}", "INFO")
            bf16_gguf_path = temp_processing_dir / f"{model_name_for_files}.bf16.gguf"
            source_model_dir_for_conversion = ""
            if model_source == "HF Hub":
                hf_download_subdir = temp_processing_dir / "hf_model_download"
                hf_download_subdir.mkdir(exist_ok=True)
                # Use a subfolder within hf_download_subdir to avoid issues if model_id has slashes
                model_download_target_dir = hf_download_subdir / model_id.replace("/", "_")
                model_download_target_dir.mkdir(exist_ok=True)
                source_model_dir_for_conversion = str(model_download_target_dir)

                log_fn(f"Downloading HF model {model_id} to {source_model_dir_for_conversion}...", "INFO")
                downloader_api = HfApi(token=hf_token_gguf) if hf_token_gguf else HfApi()
                downloader_api.snapshot_download(
                    repo_id=model_id, local_dir=source_model_dir_for_conversion,
                    allow_patterns=["*.md", "*.json", "*.bin", "*.safetensors", "*.model", "tokenizer*", "vocab*", "merges.txt", "config.json"]
                )
                log_fn(f"HF Model {model_id} downloaded.", "INFO")
            else: source_model_dir_for_conversion = local_model_path

            if not Path(source_model_dir_for_conversion).exists():
                raise GGUFConversionError(f"Source model directory for conversion does not exist: {source_model_dir_for_conversion}")


            if not Path(CONVERSION_SCRIPT).is_file():
                raise GGUFConversionError(f"GGUF conversion script not found: {CONVERSION_SCRIPT}")
            convert_cmd = [
                "python", CONVERSION_SCRIPT, str(source_model_dir_for_conversion),
                "--no-lazy", "--outtype", "bf16", "--outfile", str(bf16_gguf_path)
            ]
            env_for_py_script = os.environ.copy(); env_for_py_script["PYTHONUNBUFFERED"] = "1"
            _run_gguf_command_with_logging(convert_cmd, log_fn, env=env_for_py_script, operation_name="BF16 GGUF Conversion")
            log_fn(f"Converted to BF16 GGUF: {bf16_gguf_path}", "INFO")
            all_saved_files_info_local.append({"type": "bf16_gguf", "path": str(bf16_gguf_path), "name": bf16_gguf_path.name})

            imatrix_data_path_str = None
            if use_imatrix and quant_methods_to_run:
                imatrix_data_path = temp_processing_dir / "imatrix.dat"
                imatrix_data_path_str = str(imatrix_data_path)
                train_data_source_path = "llama.cpp/groups_merged.txt"
                if train_data_file:
                    # Ensure train_data_file is a path string
                    train_data_source_path = train_data_file.name if hasattr(train_data_file, 'name') and train_data_file.name else str(train_data_file)
                if not Path(train_data_source_path).exists():
                    raise GGUFConversionError(f"Training data for Imatrix not found: {train_data_source_path}")
                generate_importance_matrix(str(bf16_gguf_path), train_data_source_path, imatrix_data_path_str, log_fn)
                log_fn(f"Importance matrix generated: {imatrix_data_path_str}", "INFO")

            for method in quant_methods_to_run:
                quant_name_base = f"{model_name_for_files}-{method.lower().replace('_','-')}{'-imat' if use_imatrix else ''}.gguf"
                quant_output_path_temp = temp_processing_dir / quant_name_base
                quant_cmd = [LLAMA_QUANTIZE_PATH]
                if use_imatrix and imatrix_data_path_str: quant_cmd.extend(["--imatrix", imatrix_data_path_str])
                quant_cmd.extend([str(bf16_gguf_path), str(quant_output_path_temp), method])
                _run_gguf_command_with_logging(quant_cmd, log_fn, operation_name=f"Quantization ({method})", timeout=7200)
                log_fn(f"Successfully quantized ({method}) to {quant_output_path_temp}", "INFO")
                all_saved_files_info_local.append({"type": "quantized_gguf", "method": method, "path": str(quant_output_path_temp), "name": quant_name_base})

                if split_model:
                    log_fn(f"Splitting quantized model: {quant_name_base}", "INFO")
                    _, sharded_basenames = split_upload_model(
                        str(quant_output_path_temp), str(temp_processing_dir),
                        new_repo_id_gguf_for_upload if upload_to_hf else None, hf_token_gguf,
                        log_fn, split_max_tensors, split_max_size
                    )
                    if upload_to_hf and new_repo_id_gguf_for_upload:
                        log_fn(f"Sharded {quant_name_base} uploaded to {new_repo_id_gguf_for_upload}: {sharded_basenames}", "INFO")
                    if local_output_path:
                        Path(local_output_path).mkdir(parents=True, exist_ok=True)
                        for shard_basename in sharded_basenames:
                            dest_shard_path = Path(local_output_path) / shard_basename
                            shutil.copy2(temp_processing_dir / shard_basename, dest_shard_path)
                            log_fn(f"Sharded file {shard_basename} saved to: {dest_shard_path}", "INFO")
                    if quant_output_path_temp.exists(): quant_output_path_temp.unlink()
                    continue

                if upload_to_hf and hf_api_for_gguf_upload and new_repo_id_gguf_for_upload:
                    log_fn(f"Uploading {quant_name_base} to {new_repo_id_gguf_for_upload}...", "INFO")
                    hf_api_for_gguf_upload.upload_file(
                        path_or_fileobj=str(quant_output_path_temp), path_in_repo=quant_name_base, repo_id=new_repo_id_gguf_for_upload
                    )
                    log_fn(f"Uploaded {quant_name_base} to {new_repo_id_gguf_for_upload}", "INFO")
                if local_output_path:
                    Path(local_output_path).mkdir(parents=True, exist_ok=True)
                    dest_quant_path = Path(local_output_path) / quant_name_base
                    shutil.copy2(str(quant_output_path_temp), dest_quant_path)
                    log_fn(f"Quantized file {quant_name_base} saved to: {dest_quant_path}", "INFO")

            # Handle BF16 only if no other quants were run, or if explicitly needed later
            if not quant_methods_to_run and bf16_gguf_path.exists():
                if upload_to_hf and hf_api_for_gguf_upload and new_repo_id_gguf_for_upload:
                    log_fn(f"Uploading BF16 GGUF {bf16_gguf_path.name} to {new_repo_id_gguf_for_upload}...", "INFO")
                    hf_api_for_gguf_upload.upload_file(
                        path_or_fileobj=str(bf16_gguf_path), path_in_repo=bf16_gguf_path.name, repo_id=new_repo_id_gguf_for_upload
                    )
                    log_fn(f"Uploaded BF16 GGUF to {new_repo_id_gguf_for_upload}", "INFO")
                if local_output_path:
                    Path(local_output_path).mkdir(parents=True, exist_ok=True)
                    dest_bf16_path = Path(local_output_path) / bf16_gguf_path.name
                    shutil.copy2(str(bf16_gguf_path), dest_bf16_path)
                    log_fn(f"BF16 GGUF saved to: {dest_bf16_path}", "INFO")

            if upload_to_hf and hf_api_for_gguf_upload and new_repo_id_gguf_for_upload:
                card_content = f"# {new_repo_id_gguf_for_upload}\nGGUF quants of `{model_id or model_name_for_files}`.\n**Methods:** {', '.join(quant_methods_to_run or ['bf16'])}\n**iMatrix:** {'Yes' if use_imatrix and quant_methods_to_run else 'No'}"
                orig_card_content = ""
                if model_source == "HF Hub" and model_id:
                    try: orig_card_content = ModelCard.load(model_id, token=hf_token_gguf).content
                    except Exception: log_fn(f"Could not load original model card for {model_id}.", "WARNING")
                final_card_content = f"{card_content}\n\n---\n\n{orig_card_content}" if orig_card_content else card_content
                final_tags = ["llama-cpp", "gguf"]
                try:
                    ModelCard(final_card_content).push_to_hub(new_repo_id_gguf_for_upload, token=hf_token_gguf, tags=final_tags)
                    log_fn(f"README.md pushed to {new_repo_id_gguf_for_upload}", "INFO")
                except Exception as e_readme:
                    log_fn(f"Error pushing README.md: {str(e_readme)}", "ERROR")

            summary_parts = ["<h1>✅ GGUF Processing Summary</h1><ul>"]
            processed_files_display = []
            if bf16_gguf_path.exists(): # This is the one in temp_processing_dir
                processed_files_display.append(bf16_gguf_path.name + " (BF16)")
            for method in quant_methods_to_run:
                quant_name_base = f"{model_name_for_files}-{method.lower().replace('_','-')}{'-imat' if use_imatrix else ''}.gguf"
                processed_files_display.append(quant_name_base + f" ({method})")

            if processed_files_display:
                summary_parts.append("<li>Files processed (base names):</li><ul>")
                for name_desc in processed_files_display[:7]: # Show a few
                    summary_parts.append(f"<li>{escape(name_desc)}</li>")
                if len(processed_files_display) > 7: summary_parts.append("<li>...and more.</li>")
                summary_parts.append("</ul>")

            if upload_to_hf and new_repo_id_gguf_for_upload:
                summary_parts.append(f"<li>🔗 <a href='https://huggingface.co/{new_repo_id_gguf_for_upload}' target='_blank'>View GGUF Repository</a></li>")
            if local_output_path and Path(local_output_path).exists() and any(Path(local_output_path).iterdir()):
                summary_parts.append(f"<li>Files saved to local directory: {Path(local_output_path).resolve()}</li>")
            summary_parts.append("</ul>")
            result_container['final_html'] = "".join(summary_parts)
            result_container['image_path'] = "llama.png" # Placeholder, consider generating/finding a relevant image

    except GGUFConversionError as e:
        log_fn(f"GGUFConversionError: {str(e)}", "ERROR")
        result_container['error_msg'] = str(e)
        result_container['final_html'] = f'<h1>❌ GGUF PROCESSING ERROR</h1><p style="white-space:pre-wrap;">{escape(str(e))}</p>'
        result_container['image_path'] = "error.png"
    except Exception as e:
        import traceback
        tb_str = traceback.format_exc()
        log_fn(f"Unexpected error in process_gguf_conversion: {str(e)}\nTraceback: {tb_str}", "CRITICAL_ERROR")
        result_container['error_msg'] = f"Unexpected error: {str(e)}"
        result_container['final_html'] = f'<h1>❌ UNEXPECTED ERROR</h1><p style="white-space:pre-wrap;">{escape(str(e))}<br/><br/>Traceback:<br/>{escape(tb_str)}</p>'
        result_container['image_path'] = "error.png"