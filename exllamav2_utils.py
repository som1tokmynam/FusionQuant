import os
import subprocess
import shutil
import tempfile
import threading
from pathlib import Path
from textwrap import dedent
from typing import List, Optional, Callable, Dict, Any
from huggingface_hub import HfApi, ModelCard, create_repo, whoami, snapshot_download

# --- Custom Exception ---
class Exllamav2Error(Exception):
    """Custom exception for Exllamav2 errors."""
    pass

# --- Path to the cloned exllamav2 repository and convert.py ---
EXLLAMAV2_REPO_PATH_STR = os.environ.get("EXLLAMAV2_REPO_DIR", "/home/builder/app/exllamav2_repo")
CONVERT_SCRIPT_PATH = Path(EXLLAMAV2_REPO_PATH_STR) / "convert.py"

# --- Helper to create a robust model card ---
def _create_exl2_model_card_content(
    repo_id: str,
    base_model_identifier: str,
    bits: float,
    head_bits: int,
    calibration_mode: str,
    is_measurement_reused: bool,
    user_measurement_file: Optional[str],
    cal_dataset: Optional[str],
    cal_rows: int,
    is_gated: bool
) -> str:
    """Creates the Markdown content for the EXL2 model card."""
    
    # Safely build YAML frontmatter
    card_metadata = ["---", "license: mit"]
    if is_gated:
        card_metadata.append("gated: true")
    card_metadata.extend([
        "tags:",
        "  - exllamav2",
        "  - quantized",
        "library_name: exllamav2",
        "---"
    ])
    card_metadata_str = "\n".join(card_metadata)

    # Determine the calibration source string
    calibration_source_str = ""
    if calibration_mode == "use_existing_measurement_file":
        calibration_source_str = f"User-provided `measurement.json`: `{user_measurement_file}`."
    elif is_measurement_reused:
        calibration_source_str = f"Reused `measurement.json` from initial calibration pass (dataset: `{cal_dataset}`, {cal_rows} rows)."
    elif calibration_mode == "calibrate_with_dataset":
        calibration_source_str = f"Dataset: `{cal_dataset}` ({cal_rows} rows)."
    else:
        calibration_source_str = "Measurement derived from model weights (no explicit dataset calibration)."

    # Build the main body of the card
    model_card_title = repo_id.split('/')[-1]
    card_body = dedent(f"""
    # {model_card_title}

    This is an EXL2 quantized model of `{base_model_identifier}`.

    ## Quantization Details
    - **Bits per weight (bpw):** {bits}
    - **Head Bits:** {head_bits if head_bits > 0 else 'N/A'}
    - **Calibration Source:** {calibration_source_str}

    This model was quantized using the [exllamav2 library](https://github.com/turboderp/exllamav2).
    """)

    return f"{card_metadata_str}\n{card_body}"


# --- Helper to run shell commands ---
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
    if cwd: log_fn(f"Working directory: {cwd}", "DEBUG")

    effective_env = os.environ.copy()
    if env: effective_env.update(env)
    effective_env["PYTHONUNBUFFERED"] = "1"
    
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
        log_fn(f"{operation_name} timed out after {timeout} seconds. Terminating...", "ERROR")
        process.terminate()
        try: process.wait(timeout=30)
        except subprocess.TimeoutExpired: process.kill(); process.wait()
        thread.join(timeout=10)
        raise Exllamav2Error(f"{operation_name} timed out: {' '.join(cmd_list)}")

    thread.join(timeout=30)

    if process.returncode != 0:
        raise Exllamav2Error(f"{operation_name} failed with exit code {process.returncode}: {' '.join(cmd_list)}")
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
    hf_repo_private_exl2: bool = False,
    hf_repo_gated_exl2: bool = False,
    user_specified_local_output_path: bool = True, 
    temp_dir_base: Optional[str] = None,
):
    result_container['final_status_messages_list'] = []
    result_container['error_message'] = None 

    original_model_name_part = Path(hf_model_id_or_path).name

    active_measurement_file_for_reuse: Optional[Path] = None
    job_level_temp_measurement_dir: Optional[Path] = None
    downloaded_model_dir_for_job: Optional[Path] = None

    if not CONVERT_SCRIPT_PATH.is_file():
        raise Exllamav2Error(f"ExLlamaV2 convert.py not found: {CONVERT_SCRIPT_PATH}")

    if calibration_mode == "use_existing_measurement_file":
        if input_measurement_file_path and Path(input_measurement_file_path).is_file():
            active_measurement_file_for_reuse = Path(input_measurement_file_path)
            log_fn(f"Using user-provided measurement file: {active_measurement_file_for_reuse}", "INFO")
        else:
            raise Exllamav2Error(f"Measurement file not found: {input_measurement_file_path}")

    def set_permissive_umask():
        os.umask(0o000)
        
    try:
        log_fn(f"Starting EXL2 quantization for: {hf_model_id_or_path}", "INFO")
        effective_temp_root_for_job = Path(temp_dir_base if temp_dir_base else tempfile.gettempdir())
        hf_cache_path_for_job = effective_temp_root_for_job / "huggingface_cache"
        job_level_temp_measurement_dir = effective_temp_root_for_job / f"job_measurement_{original_model_name_part}"
        
        for path in [effective_temp_root_for_job, hf_cache_path_for_job, job_level_temp_measurement_dir]:
            path.mkdir(parents=True, exist_ok=True)

        subprocess_env = os.environ.copy()
        subprocess_env["HF_HOME"] = str(hf_cache_path_for_job)
        if hf_token_exl2: subprocess_env["HF_TOKEN"] = hf_token_exl2

        actual_model_input_path_for_convert_str = hf_model_id_or_path 
        if model_source_type == "HF Hub":
            log_fn(f"Downloading HF model: {hf_model_id_or_path}", "INFO")
            downloaded_model_dir_for_job = effective_temp_root_for_job / "downloaded_hf_model"
            try:
                actual_model_input_path_for_convert_str = snapshot_download(
                    repo_id=hf_model_id_or_path, local_dir=str(downloaded_model_dir_for_job),
                    local_dir_use_symlinks=False, token=hf_token_exl2,
                    cache_dir=str(hf_cache_path_for_job / "hub"))
                log_fn(f"Model downloaded to: {actual_model_input_path_for_convert_str}", "INFO")
            except Exception as e_download:
                raise Exllamav2Error(f"Failed to download HF model '{hf_model_id_or_path}': {e_download}")
        else: # Local Path or Merged
             actual_model_input_path_for_convert_str = str(Path(hf_model_id_or_path).resolve())

        username_for_hf = whoami(token=hf_token_exl2).get('name') if upload_to_hf_exl2 and hf_token_exl2 else None

        for i, current_bits in enumerate(bits_list):
            log_fn(f"--- Processing for bitrate: {current_bits} ---", "INFO")
            op_temp_dir = tempfile.TemporaryDirectory(prefix="exl2_op_", dir=effective_temp_root_for_job)
            intermediate_exl2_output_dir = Path(op_temp_dir.name) / "model_out"
            intermediate_exl2_output_dir.mkdir()

            cmd = ["python3", str(CONVERT_SCRIPT_PATH),
                   "-i", str(actual_model_input_path_for_convert_str),
                   "-o", str(intermediate_exl2_output_dir),
                   "-b", str(current_bits)]
            if head_bits > 0: cmd.extend(["-hb", str(head_bits)])
            
            if active_measurement_file_for_reuse and active_measurement_file_for_reuse.is_file():
                cmd.extend(["-m", str(active_measurement_file_for_reuse)])
            elif calibration_mode == "calibrate_with_dataset":
                if not calibration_dataset_path_or_hf_name:
                    raise Exllamav2Error("Calibration dataset required.")
                cmd.extend(["-c", calibration_dataset_path_or_hf_name, "-r", str(calibration_rows)])

            _run_exllamav2_command_with_logging(
                cmd, log_fn,
                operation_name=f"Exllamav2 Quant ({current_bits} bits)",
                cwd=EXLLAMAV2_REPO_PATH_STR,
                env=subprocess_env,
                preexec_fn=None if os.name == 'nt' else set_permissive_umask
            )

            if not any(intermediate_exl2_output_dir.iterdir()):
                raise Exllamav2Error(f"Conversion for {current_bits} bits produced no output.")
            
            if calibration_mode == "calibrate_with_dataset" and not active_measurement_file_for_reuse:
                potential_measurement_file = intermediate_exl2_output_dir / "measurement.json"
                if potential_measurement_file.is_file():
                    active_measurement_file_for_reuse = job_level_temp_measurement_dir / "measurement.json"
                    shutil.copy2(potential_measurement_file, active_measurement_file_for_reuse)
                    log_fn(f"Measurement file saved for reuse: {active_measurement_file_for_reuse}", "INFO")

            quant_suffix = f"{str(current_bits).replace('.', 'p')}bpw" + (f"_H{head_bits}" if head_bits > 0 else "")
            final_local_output_path = Path(output_exl2_model_dir_base) / f"{original_model_name_part}-{quant_suffix}-exl2"
            shutil.copytree(intermediate_exl2_output_dir, final_local_output_path, dirs_exist_ok=True)
            log_fn(f"EXL2 model ({current_bits} bits) created at: {final_local_output_path}", "INFO")

            current_status_msg = f"✅ **{current_bits} bpw:** Success"
            
            if upload_to_hf_exl2 and hf_token_exl2:
                repo_base = hf_repo_name_base_exl2 or original_model_name_part
                repo_id = f"{username_for_hf}/{repo_base}-{quant_suffix}-exl2" if username_for_hf else f"{repo_base}-{quant_suffix}-exl2"
                
                try:
                    repo_url_obj = create_repo(repo_id, token=hf_token_exl2, private=hf_repo_private_exl2, exist_ok=True)
                    repo_url = str(repo_url_obj) # Get string representation for use in logs/links
                    
                    for filename in ["special_tokens_map.json", "tokenizer.json", "tokenizer_config.json"]:
                        if (Path(actual_model_input_path_for_convert_str) / filename).exists():
                            shutil.copy2(Path(actual_model_input_path_for_convert_str) / filename, final_local_output_path)
                    
                    HfApi(token=hf_token_exl2).upload_folder(
                        folder_path=str(final_local_output_path), repo_id=repo_id,
                        commit_message=f"Add EXL2 {current_bits}-bit model",
                        ignore_patterns=["out_tensor/*", "cal_data.safetensors"]
                    )
                    
                    is_reused = i > 0 and calibration_mode == "calibrate_with_dataset"
                    card_content = _create_exl2_model_card_content(
                        repo_id=repo_id,
                        base_model_identifier=hf_model_id_or_path,
                        bits=current_bits, head_bits=head_bits,
                        calibration_mode=calibration_mode,
                        is_measurement_reused=is_reused,
                        user_measurement_file=input_measurement_file_path,
                        cal_dataset=calibration_dataset_path_or_hf_name,
                        cal_rows=calibration_rows,
                        is_gated=hf_repo_gated_exl2
                    )
                    ModelCard(card_content).push_to_hub(repo_id, token=hf_token_exl2)
                    
                    access = "gated" if hf_repo_gated_exl2 else ("private" if hf_repo_private_exl2 else "public")
                    current_status_msg += f" | HF ({access}): <a href='{repo_url}' target='_blank'>{repo_id}</a>"
                    
                    if not user_specified_local_output_path:
                        shutil.rmtree(final_local_output_path)
                        current_status_msg += " | Local copy deleted."
                    
                except Exception as e_hf:
                    current_status_msg += f" | HF Upload Error: {str(e_hf)[:100]}..."
                    current_error = result_container.get('error_message') or "" # Safely handle None
                    result_container['error_message'] = (current_error + f"\nUpload failed for {repo_id}: {e_hf}").strip()
            
            result_container['final_status_messages_list'].append(current_status_msg)
            op_temp_dir.cleanup()

    except (Exllamav2Error, Exception) as e:
        import traceback
        log_fn(f"Error in EXL2 process: {e}\n{traceback.format_exc()}", "CRITICAL_ERROR")
        result_container['error_message'] = str(e)
    finally:
        for temp_path in [job_level_temp_measurement_dir, hf_cache_path_for_job, downloaded_model_dir_for_job]:
            if temp_path and temp_path.exists():
                shutil.rmtree(temp_path, ignore_errors=True)