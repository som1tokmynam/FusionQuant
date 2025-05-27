import os
import subprocess
import shutil
import tempfile
import threading
from pathlib import Path
from typing import List, Optional, Callable, Dict, Any
from huggingface_hub import HfApi, ModelCard, create_repo, whoami, snapshot_download
import importlib.util

# --- Custom Exception ---
class Exllamav2Error(Exception):
    """Custom exception for Exllamav2 errors."""
    pass

# --- Path to the cloned exllamav2 repository and convert.py ---
EXLLAMAV2_REPO_PATH_STR = os.environ.get("EXLLAMAV2_REPO_DIR", "/home/builder/app/exllamav2_repo")
CONVERT_SCRIPT_PATH = Path(EXLLAMAV2_REPO_PATH_STR) / "convert.py"

# --- Helper Functions ---
def _run_exllamav2_command_with_logging(
    cmd_list: List[str],
    log_fn: Callable[[str, str], None],
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    timeout: int = 14400, 
    operation_name: str = "Exllamav2 operation",
    preexec_fn: Optional[Callable[[], None]] = None
):
    log_fn(f"Running {operation_name}: {' '.join(cmd_list)}", "INFO")
    if cwd:
        log_fn(f"Working directory: {cwd}", "DEBUG")

    effective_env = os.environ.copy()
    if env:
        effective_env.update(env)

    effective_env["PYTHONUNBUFFERED"] = "1"
    if "HF_HOME" in effective_env:
        log_fn(f"Subprocess {operation_name} using HF_HOME: {effective_env['HF_HOME']}", "DEBUG")
    if "HF_TOKEN" in effective_env and effective_env.get("HF_TOKEN"):
        log_fn(f"Subprocess {operation_name} using passed HF_TOKEN.", "DEBUG")
    else:
        log_fn(f"Subprocess {operation_name} *not* using a passed HF_TOKEN (either not provided or empty).", "DEBUG")

    process = subprocess.Popen(
        cmd_list,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
        cwd=cwd,
        env=effective_env,
        errors='replace',
        preexec_fn=preexec_fn
    )

    def read_stream():
        try:
            for line in iter(process.stdout.readline, ''): 
                if line:
                    log_fn(line.rstrip('\n\r'), "INFO") 
        except Exception as e_read:
            log_fn(f"Error reading stream for {operation_name}: {e_read}", "ERROR")
        finally:
            if process.stdout: 
                process.stdout.close() 

    thread = threading.Thread(target=read_stream)
    thread.daemon = True
    thread.start()

    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        log_fn(f"{operation_name} timed out after {timeout} seconds. Terminating process...", "ERROR")
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            log_fn(f"{operation_name} did not terminate gracefully. Killing process...", "ERROR")
            process.kill()
            process.wait()
        thread.join(timeout=10)
        error_msg = f"{operation_name} timed out: {' '.join(cmd_list)}"
        raise Exllamav2Error(error_msg)

    thread.join(timeout=30)

    if process.returncode != 0:
        error_msg = f"{operation_name} failed with exit code {process.returncode}: {' '.join(cmd_list)}"
        log_fn(error_msg, "ERROR")
        raise Exllamav2Error(error_msg)

    log_fn(f"{operation_name} completed successfully.", "INFO")


def process_exllamav2_quantization(
    model_source_type: str,
    hf_model_id_or_path: str,
    output_exl2_model_dir_base: str, 
    bits_list: List[float],
    head_bits: int,
    log_fn: Callable[[str, str], None],
    result_container: Dict[str, Any],
    calibration_mode: str = "none",
    calibration_dataset_path_or_hf_name: Optional[str] = None,
    calibration_rows: int = 100,
    input_measurement_file_path: Optional[str] = None,
    hf_token_exl2: Optional[str] = None,
    upload_to_hf_exl2: bool = False,
    hf_repo_name_base_exl2: Optional[str] = None,
    hf_repo_private_exl2: bool = False, # Retained
    hf_repo_gated_exl2: bool = False,   # New flag for gated repo
    user_specified_local_output_path: bool = True, 
    temp_dir_base: Optional[str] = None,
):
    result_container['final_status_messages_list'] = []
    result_container['error_message'] = None 

    if model_source_type == "HF Hub":
        original_model_name_part = Path(hf_model_id_or_path).name
    else:
        original_model_name_part = Path(hf_model_id_or_path).name
        if not original_model_name_part:
             original_model_name_part = Path(hf_model_id_or_path).parent.name

    active_measurement_file_for_reuse: Optional[Path] = None
    job_level_temp_measurement_dir: Optional[Path] = None
    downloaded_model_dir_for_job: Optional[Path] = None

    if not CONVERT_SCRIPT_PATH.is_file():
        err_msg = f"ExLlamaV2 convert.py script not found at expected location: {CONVERT_SCRIPT_PATH}. Check EXLLAMAV2_REPO_DIR env var."
        log_fn(err_msg, "CRITICAL_ERROR")
        raise Exllamav2Error(err_msg)

    if calibration_mode == "use_existing_measurement_file":
        if input_measurement_file_path and Path(input_measurement_file_path).is_file():
            active_measurement_file_for_reuse = Path(input_measurement_file_path)
            log_fn(f"Using user-provided measurement file for all bitrates: {active_measurement_file_for_reuse}", "INFO")
        else:
            raise Exllamav2Error(f"Calibration mode is 'use_existing_measurement_file' but path is invalid or not found: {input_measurement_file_path}")

    def set_permissive_umask():
        os.umask(0o000)
        log_fn("Umask set to 0o000 for subprocess.", "DEBUG")
        
    try:
        log_fn(f"Starting EXL2 quantization for input: {hf_model_id_or_path}", "INFO")
        effective_temp_root_for_job = Path(temp_dir_base if temp_dir_base else tempfile.gettempdir())
        effective_temp_root_for_job.mkdir(parents=True, exist_ok=True)
        hf_cache_path_for_job = effective_temp_root_for_job / "huggingface_cache"
        hf_cache_path_for_job.mkdir(parents=True, exist_ok=True)

        subprocess_env = os.environ.copy()
        subprocess_env["HF_HOME"] = str(hf_cache_path_for_job)
        subprocess_env["TRANSFORMERS_CACHE"] = str(hf_cache_path_for_job)
        subprocess_env["HF_HUB_CACHE"] = str(hf_cache_path_for_job)
        if hf_token_exl2: subprocess_env["HF_TOKEN"] = hf_token_exl2

        job_level_temp_measurement_dir = effective_temp_root_for_job / f"job_measurement_{original_model_name_part}_{os.getpid()}"
        job_level_temp_measurement_dir.mkdir(parents=True, exist_ok=True)
        actual_model_input_path_for_convert_str = hf_model_id_or_path 

        if model_source_type == "HF Hub":
            log_fn(f"Input '{hf_model_id_or_path}' is from HF Hub. Attempting pre-download...", "INFO")
            downloaded_model_dir_for_job = effective_temp_root_for_job / "downloaded_hf_model" / original_model_name_part
            downloaded_model_dir_for_job.mkdir(parents=True, exist_ok=True)
            if not os.name == 'nt': 
                try: os.chmod(str(downloaded_model_dir_for_job), 0o777)
                except Exception as e_chmod: log_fn(f"Warning: Could not chmod downloaded_model_dir_for_job {downloaded_model_dir_for_job}: {e_chmod}", "WARNING")
            try:
                actual_model_input_path_for_convert_str = snapshot_download(
                    repo_id=hf_model_id_or_path, local_dir=str(downloaded_model_dir_for_job),
                    local_dir_use_symlinks=False, token=hf_token_exl2,
                    cache_dir=str(hf_cache_path_for_job / "hub"))
                log_fn(f"Model '{hf_model_id_or_path}' downloaded to: {actual_model_input_path_for_convert_str}", "INFO")
            except Exception as e_download:
                raise Exllamav2Error(f"Failed to download HF model '{hf_model_id_or_path}': {e_download}")
        elif model_source_type in ["Local Path", "Output from Merge Step"]:
            if not Path(hf_model_id_or_path).exists():
                raise Exllamav2Error(f"Provided local model path does not exist: {hf_model_id_or_path}")
            actual_model_input_path_for_convert_str = str(Path(hf_model_id_or_path).resolve())
            log_fn(f"Using provided local model path: {actual_model_input_path_for_convert_str}", "INFO")

        username_for_hf = None
        if upload_to_hf_exl2 and hf_token_exl2:
            try:
                user_info = whoami(token=hf_token_exl2)
                username_for_hf = user_info.get('name')
                if not username_for_hf:
                    log_fn("Could not determine HF username. Upload may fail if repo name is not fully qualified.", "WARNING")
            except Exception as e_whoami:
                log_fn(f"Failed to get HF user info: {e_whoami}. Upload may fail.", "WARNING")


        for i, current_bits in enumerate(bits_list):
            log_fn(f"--- Processing for bitrate: {current_bits} ---", "INFO")
            op_temp_dir_for_bitrate = effective_temp_root_for_job / f"exl2_quant_op_{str(current_bits).replace('.', 'p')}_{os.urandom(4).hex()}"
            op_temp_dir_for_bitrate.mkdir(parents=True, exist_ok=True)
            if not os.name == 'nt':
                try: os.chmod(str(op_temp_dir_for_bitrate), 0o777)
                except Exception as e: log_fn(f"Failed to chmod op_temp_dir_for_bitrate: {e}", "WARNING")

            intermediate_exl2_output_dir = op_temp_dir_for_bitrate / "exl2_model_out_temp"
            intermediate_exl2_output_dir.mkdir(parents=True, exist_ok=True)
            if not os.name == 'nt':
                try: os.chmod(str(intermediate_exl2_output_dir), 0o777)
                except Exception as e: log_fn(f"Failed to chmod intermediate_exl2_output_dir: {e}", "WARNING")

            cmd = ["python3", str(CONVERT_SCRIPT_PATH)]
            cmd.extend(["-i", str(actual_model_input_path_for_convert_str)])
            cmd.extend(["-o", str(intermediate_exl2_output_dir)])
            cmd.extend(["-b", str(current_bits)])
            if head_bits > 0: cmd.extend(["-hb", str(head_bits)])
            
            if active_measurement_file_for_reuse and active_measurement_file_for_reuse.is_file():
                cmd.extend(["-m", str(active_measurement_file_for_reuse)])
            elif calibration_mode == "calibrate_with_dataset":
                if not calibration_dataset_path_or_hf_name:
                    raise Exllamav2Error("Calibration dataset is required for 'calibrate_with_dataset' mode.")
                cmd.extend(["-c", calibration_dataset_path_or_hf_name])
                cmd.extend(["-r", str(calibration_rows)])

            _run_exllamav2_command_with_logging(
                cmd, log_fn,
                operation_name=f"Exllamav2 Quant ({current_bits} bits)",
                cwd=EXLLAMAV2_REPO_PATH_STR,
                env=subprocess_env,
                preexec_fn=set_permissive_umask
            )

            if not intermediate_exl2_output_dir.exists() or not any(intermediate_exl2_output_dir.iterdir()):
                raise Exllamav2Error(f"Exllamav2 conversion for {current_bits} bits did not produce output in {intermediate_exl2_output_dir}")
            
            if calibration_mode == "calibrate_with_dataset" and not active_measurement_file_for_reuse:
                potential_measurement_file = intermediate_exl2_output_dir / "measurement.json"
                if potential_measurement_file.is_file():
                    reused_measurement_target_path = job_level_temp_measurement_dir / f"measurement_{original_model_name_part}.json"
                    shutil.copy2(potential_measurement_file, reused_measurement_target_path)
                    active_measurement_file_for_reuse = reused_measurement_target_path
                    log_fn(f"Measurement file '{potential_measurement_file.name}' saved for reuse: {active_measurement_file_for_reuse}", "INFO")
                else:
                    log_fn(f"WARNING: Expected measurement.json not found in {intermediate_exl2_output_dir} after calibration.", "WARNING")

            quant_suffix_for_path = f"{original_model_name_part}_{str(current_bits).replace('.', 'p')}bpw"
            if head_bits > 0:
                quant_suffix_for_path += f"_H{head_bits}"
            quant_suffix_for_path += "_exl2"
            final_local_output_path_this_bitrate = Path(output_exl2_model_dir_base) / quant_suffix_for_path
            final_local_output_path_this_bitrate.parent.mkdir(parents=True, exist_ok=True)
            if final_local_output_path_this_bitrate.exists():
                shutil.rmtree(final_local_output_path_this_bitrate)
            shutil.copytree(intermediate_exl2_output_dir, final_local_output_path_this_bitrate)
            log_fn(f"EXL2 model ({current_bits} bits) copied to: {final_local_output_path_this_bitrate}", "INFO")

            current_status_msg = f"✅ **{current_bits} bits:** Conversion successful."
            local_path_kept_and_exists = True 

            if upload_to_hf_exl2 and hf_token_exl2:
                api_hf = HfApi(token=hf_token_exl2)
                if not hf_repo_name_base_exl2 or not hf_repo_name_base_exl2.strip():
                    repo_build_base_name = original_model_name_part
                else:
                    repo_build_base_name = hf_repo_name_base_exl2
                
                bits_str_display_hf = f"{current_bits:.1f}" if current_bits == float(int(current_bits)) else str(current_bits)
                quant_specific_hf_suffix = f"-{bits_str_display_hf}bpw"
                if head_bits > 0:
                    quant_specific_hf_suffix += f"-H{head_bits}"
                
                repo_id_this_bitrate: Optional[str] = None
                if '/' not in repo_build_base_name:
                    if not username_for_hf:
                        err_msg_repo = "Cannot form full HF repo ID: Username not found and repo base name has no namespace."
                        log_fn(err_msg_repo, "ERROR")
                        current_error_val_repo = result_container.get('error_message')
                        if current_error_val_repo is None: current_error_val_repo = ""
                        if err_msg_repo not in current_error_val_repo:
                            result_container['error_message'] = (current_error_val_repo + "\n" + err_msg_repo).strip()
                        current_status_msg += " | HF Upload Failed (username missing)"
                    else:
                        final_repo_model_name_segment = f"{repo_build_base_name}-EXL2"
                        repo_id_this_bitrate = f"{username_for_hf}/{final_repo_model_name_segment}{quant_specific_hf_suffix}"
                else:
                    namespace, name_part_from_user = repo_build_base_name.split('/', 1)
                    final_repo_model_name_segment = f"{name_part_from_user}-EXL2" if not name_part_from_user.endswith("-EXL2") else name_part_from_user
                    repo_id_this_bitrate = f"{namespace}/{final_repo_model_name_segment}{quant_specific_hf_suffix}"

                if repo_id_this_bitrate:
                    try:
                        create_repo_private_flag = hf_repo_private_exl2
                        if hf_repo_gated_exl2: # Gated implies public creation
                            create_repo_private_flag = False

                        repo_url_obj_this_bitrate = create_repo(
                            repo_id=repo_id_this_bitrate, token=hf_token_exl2,
                            private=create_repo_private_flag, exist_ok=True, repo_type="model"
                        )
                        repo_url_this_bitrate = repo_url_obj_this_bitrate.repo_url if hasattr(repo_url_obj_this_bitrate, 'repo_url') else str(repo_url_obj_this_bitrate)
                        
                        source_model_dir_for_configs = Path(actual_model_input_path_for_convert_str)
                        target_upload_dir = Path(final_local_output_path_this_bitrate)
                        essential_files_to_copy = [
                            "special_tokens_map.json", "tokenizer.json",
                            "tokenizer_config.json", "generation_config.json"
                        ]
                        if not (target_upload_dir / "config.json").exists():
                            log_fn(f"CRITICAL WARNING: 'config.json' not found in EXL2 output '{target_upload_dir}'. Model may be unusable.", "ERROR")
                        for filename in essential_files_to_copy:
                            source_file = source_model_dir_for_configs / filename
                            target_file = target_upload_dir / filename
                            if source_file.exists() and not target_file.exists():
                                try:
                                    shutil.copy2(source_file, target_file)
                                    log_fn(f"Copied missing '{filename}' to EXL2 output for upload.", "INFO")
                                except Exception as e_copy:
                                    log_fn(f"Warning: Failed to copy '{filename}': {e_copy}", "WARNING")
                            elif not source_file.exists() and filename != "generation_config.json":
                                log_fn(f"Warning: Source file '{filename}' not in '{source_model_dir_for_configs}'. Cannot ensure for upload.", "WARNING")
                        
                        log_fn(f"Contents of upload folder '{target_upload_dir}' before HF upload:", "INFO")
                        for item_path in sorted(target_upload_dir.rglob('*')): 
                            relative_item_path = item_path.relative_to(target_upload_dir)
                            log_fn(f"  - {relative_item_path} {'(dir)' if item_path.is_dir() else ''}", "INFO")

                        ignore_patterns = [
                            "out_tensor/*", "out_tensor", 
                            "cal_data.safetensors",
                            "hidden_states.safetensors",
                            "job_new.json", 
                            ".gitattributes"
                        ]
                        log_fn(f"Uploading to {repo_id_this_bitrate} (ignore patterns: {ignore_patterns})...", "INFO")
                        api_hf.upload_folder(
                            folder_path=str(final_local_output_path_this_bitrate),
                            repo_id=repo_id_this_bitrate, repo_type="model",
                            commit_message=f"Add EXL2 {current_bits}-bit (head_bits: {head_bits if head_bits > 0 else 'N/A'}) quantized model",
                            ignore_patterns=ignore_patterns 
                        )
                        
                        card_metadata_list = ["---", "license: mit"]
                        if hf_repo_gated_exl2:
                            card_metadata_list.append("gated: true")
                        card_metadata_list.extend([
                            "tags:",
                            "  - exllamav2",
                            "  - quantized",
                            "library_name: exllamav2",
                            "---"
                        ])
                        card_metadata_str = "\n".join(card_metadata_list) + "\n"
                        
                        model_card_title = repo_id_this_bitrate.split('/')[-1]
                        card_body = (f"# {model_card_title}\n\n"
                                     f"EXL2 quantized model of `{hf_repo_name_base_exl2 or hf_model_id_or_path}` (the original base model).\n\n"
                                     f"## Quantization Details\n"
                                     f"- **Bits per weight (bpw):** {current_bits}\n")
                        if head_bits > 0: card_body += f"- **Head Bits:** {head_bits}\n"
                        card_body += f"- **Calibration Source:** "
                        is_reused_measurement = (active_measurement_file_for_reuse and 
                                                 active_measurement_file_for_reuse.name == f"measurement_{original_model_name_part}.json" and
                                                 active_measurement_file_for_reuse.parent == job_level_temp_measurement_dir)
                        if calibration_mode == "use_existing_measurement_file":
                            card_body += f"User-provided `measurement.json`: `{input_measurement_file_path}`.\n"
                        elif is_reused_measurement and i > 0 :
                             card_body += f"Reused `measurement.json` from initial calibration pass (dataset: `{calibration_dataset_path_or_hf_name}`, {calibration_rows} rows).\n"
                        elif calibration_mode == "calibrate_with_dataset":
                            card_body += f"Dataset: `{calibration_dataset_path_or_hf_name}` ({calibration_rows} rows).\n"
                        else:
                            card_body += "Measurement derived from model weights (no explicit dataset calibration or provided measurement for this specific quantization pass).\n"
                        card_body += f"\nQuantized using the [exllamav2 library](https://github.com/turboderp/exllamav2)."
                        card_content_this_bitrate = card_metadata_str + card_body
                        ModelCard(card_content_this_bitrate).push_to_hub(repo_id_this_bitrate, token=hf_token_exl2)
                        
                        repo_access_type_log = "private" if create_repo_private_flag else ("gated" if hf_repo_gated_exl2 else "public")
                        current_status_msg += f" | HF ({repo_access_type_log}): <a href='{repo_url_this_bitrate}' target='_blank'>{repo_id_this_bitrate}</a>"


                        if not user_specified_local_output_path: 
                            log_fn(f"HF upload successful and no specific local path provided. Deleting default local copy: {final_local_output_path_this_bitrate}", "INFO")
                            try:
                                shutil.rmtree(final_local_output_path_this_bitrate)
                                log_fn(f"Successfully deleted default local copy: {final_local_output_path_this_bitrate}", "INFO")
                                local_path_kept_and_exists = False 
                            except Exception as e_delete_local:
                                log_fn(f"Warning: Failed to delete default local copy {final_local_output_path_this_bitrate}: {e_delete_local}", "WARNING")
                    except Exception as e_hf_upload:
                        upload_err_msg = f"Error during Hugging Face upload or ModelCard push for {repo_id_this_bitrate}: {str(e_hf_upload)}"
                        log_fn(upload_err_msg, "ERROR")
                        current_error_value = result_container.get('error_message')
                        if current_error_value is None: current_error_value = "" 
                        if upload_err_msg not in current_error_value:
                            result_container['error_message'] = (current_error_value + "\n" + upload_err_msg).strip()
                        current_status_msg += f" | HF Upload/Card Error: {str(e_hf_upload)[:100]}..."
            
            if local_path_kept_and_exists:
                current_status_msg += f" | Local: `{str(final_local_output_path_this_bitrate.resolve())}`"
            elif not user_specified_local_output_path: 
                current_status_msg += " | Local copy (default path) deleted after successful HF upload."
            
            result_container['final_status_messages_list'].append(current_status_msg)

            if op_temp_dir_for_bitrate.exists():
                try: shutil.rmtree(op_temp_dir_for_bitrate)
                except Exception as e_clean_op_temp: log_fn(f"Warning: Failed to clean up bitrate-specific temp output directory {op_temp_dir_for_bitrate}: {e_clean_op_temp}", "WARNING")

    except Exllamav2Error as e:
        log_fn(f"Exllamav2Error: {str(e)}", "ERROR")
        result_container['error_message'] = str(e)
        if not result_container['final_status_messages_list']:
            result_container['final_status_messages_list'].append(f"❌ **Overall Error:** {str(e)}")
    except Exception as e:
        import traceback
        tb_str = traceback.format_exc()
        log_fn(f"Unexpected error in EXL2 quantization: {str(e)}\nTraceback:\n{tb_str}", "CRITICAL_ERROR")
        result_container['error_message'] = f"Unexpected critical error: {str(e)}"
        if not result_container['final_status_messages_list']:
            result_container['final_status_messages_list'].append(f"❌ **Critical Unexpected Error:** {str(e)}")
    finally:
        if job_level_temp_measurement_dir and job_level_temp_measurement_dir.exists():
            try: shutil.rmtree(job_level_temp_measurement_dir)
            except Exception as e_clean_measure: log_fn(f"Warning: Failed to clean job-level measurement directory {job_level_temp_measurement_dir}: {e_clean_measure}", "WARNING")
        if hf_cache_path_for_job and hf_cache_path_for_job.exists():
            try: shutil.rmtree(hf_cache_path_for_job)
            except Exception as e_clean_hf_cache: log_fn(f"Warning: Failed to clean job-level HF cache directory {hf_cache_path_for_job}: {e_clean_hf_cache}", "WARNING")
        if downloaded_model_dir_for_job and downloaded_model_dir_for_job.exists():
            try: shutil.rmtree(downloaded_model_dir_for_job)
            except Exception as e_clean_dl: log_fn(f"Warning: Failed to clean job-level downloaded model directory {downloaded_model_dir_for_job}: {e_clean_dl}", "WARNING")