import os
import pathlib
import random
import string
import tempfile
import time
import shutil
import subprocess
import threading
import signal
from typing import Iterable, List, Tuple, Optional, Callable, Any

import huggingface_hub
import torch
import yaml
from mergekit.config import MergeConfiguration
import shlex

class MergekitError(Exception):
    """Custom exception for Mergekit errors."""
    pass

def get_example_yaml_filenames_for_gr_examples(examples_dir_str: str = "examples") -> List[str]:
    """Get list of example YAML filenames for Gradio examples."""
    examples_path = pathlib.Path(examples_dir_str)
    filenames = []

    if examples_path.exists() and examples_path.is_dir():
        for yaml_file in examples_path.glob("*.yaml"):
            try:
                filenames.append(yaml_file.name)
            except Exception as e:
                print(f"Error accessing example file {yaml_file}: {e}")

    if not filenames:
        if not (examples_path / "default_example.yaml").exists():
            try:
                create_example_files(examples_dir_str)
                if (examples_path / "default_example.yaml").exists():
                    filenames.append("default_example.yaml")
            except Exception as e:
                print(f"Failed to create default example in get_example_yaml_filenames_for_gr_examples: {e}")
        elif (examples_path / "default_example.yaml").exists():
             filenames.append("default_example.yaml")

    if not filenames:
        filenames.append("default_example.yaml") # Fallback even if creation failed
    return filenames

def get_example_yaml_content(filename: str, examples_dir_str: str = "examples") -> str:
    """Get the content of a specific example YAML file."""
    examples_path = pathlib.Path(examples_dir_str)
    yaml_file_path = examples_path / filename

    if filename == "default_example.yaml" or not yaml_file_path.exists():
        # Fallback to a hardcoded default if the file doesn't exist or if "default_example.yaml" is specifically requested
        # This ensures that if create_example_files fails or examples are missing, a default is still provided.
        return """models:
  - model: cognitivecomputations/dolphin-2.6-mistral-7b-dpo-laser
    parameters:
      weight: 0.5
  - model: OpenPipe/mistral-ft-optimized-1218
    parameters:
      weight: 0.5
merge_method: linear
dtype: float16"""

    try:
        with open(yaml_file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading example file {yaml_file_path}: {e}")
        return f"# Error reading {filename}\n# {str(e)}"

def load_examples(examples_dir_str: str = "examples") -> List[List[str]]:
    """Loads example YAML file contents for Gradio UI."""
    examples_path = pathlib.Path(examples_dir_str)
    examples = []

    if examples_path.exists() and examples_path.is_dir():
        for yaml_file in examples_path.glob("*.yaml"):
            try:
                with open(yaml_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                examples.append([content]) # Gradio expects a list of lists for examples
            except Exception as e:
                print(f"Error reading example file {yaml_file}: {e}")
                examples.append([f"# Error reading {yaml_file.name}\n# {str(e)}"])

    if not examples: # Fallback if no examples were loaded
        default_example = """models:
  - model: cognitivecomputations/dolphin-2.6-mistral-7b-dpo-laser
    parameters:
      weight: 0.5
  - model: OpenPipe/mistral-ft-optimized-1218
    parameters:
      weight: 0.5
merge_method: linear
dtype: float16"""
        examples.append([default_example])
    return examples

def create_example_files(examples_dir_str: str = "examples"):
    """Create example YAML files if they don't exist."""
    examples_path = pathlib.Path(examples_dir_str)
    examples_path.mkdir(exist_ok=True) # Ensure the examples directory exists

    examples_configs = {
        "default_example.yaml": """models:
  - model: cognitivecomputations/dolphin-2.6-mistral-7b-dpo-laser
    parameters:
      weight: 0.5
  - model: OpenPipe/mistral-ft-optimized-1218
    parameters:
      weight: 0.5
merge_method: linear
dtype: float16""",
        "linear_merge.yaml": """models:
  - model: cognitivecomputations/dolphin-2.6-mistral-7b-dpo-laser
    parameters:
      weight: 0.5
  - model: OpenPipe/mistral-ft-optimized-1218
    parameters:
      weight: 0.5
merge_method: linear
dtype: float16""",
        "slerp_merge.yaml": """models:
  - model: microsoft/DialoGPT-medium
    parameters:
      weight: 0.7
  - model: microsoft/DialoGPT-large
    parameters:
      weight: 0.3
merge_method: slerp
dtype: float16
parameters:
  t: 0.5""",
        "ties_merge.yaml": """models:
  - model: huggingface/CodeBERTa-small-v1
    parameters:
      weight: 0.6
  - model: microsoft/codebert-base
    parameters:
      weight: 0.4
merge_method: ties
dtype: float16
parameters:
  density: 0.5""",
        "dare_ties_merge.yaml": """models:
  - model: teknium/OpenHermes-2.5-Mistral-7B
    parameters:
      weight: 0.5
      density: 0.53
  - model: NousResearch/Nous-Hermes-2-Mistral-7B-DPO
    parameters:
      weight: 0.5
      density: 0.53
merge_method: dare_ties
base_model: mistralai/Mistral-7B-v0.1
parameters:
  int8_mask: true
dtype: bfloat16"""
    }

    for filename, content in examples_configs.items():
        file_path = examples_path / filename
        if not file_path.exists():
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"Created example file: {file_path}")
            except Exception as e:
                print(f"Error creating example file {file_path}: {e}")

def clean_tmp_folders():
    cleaned_count = 0
    try:
        temp_dir_root = pathlib.Path(tempfile.gettempdir())
        # Define patterns for directories and files to clean
        patterns_to_clean = ['mergekit_op_*', 'merge_temp_*', '*.cache', '*.lora_cache']

        for pattern in patterns_to_clean:
            for item in temp_dir_root.glob(pattern):
                if item.is_dir():
                    try:
                        print(f"Cleaning temporary directory: {item}")
                        shutil.rmtree(item)
                        cleaned_count += 1
                    except Exception as e:
                        print(f"Failed to remove {item}: {e}")
                elif item.is_file() and (pattern.endswith(".cache") or pattern.endswith(".lora_cache")): # Ensure only specified file patterns are deleted
                    try:
                        print(f"Cleaning temporary file: {item}")
                        item.unlink()
                        cleaned_count +=1
                    except Exception as e:
                        print(f"Failed to remove {item}: {e}")
    except Exception as e:
        print(f"Error during cleanup of temp folders: {e}")
    return cleaned_count

def _run_command_with_logging(cmd_list: List[str], cwd: str, env: Optional[dict], log_fn: Callable, timeout: Optional[int] = 3600):
    log_fn(f"Running command: {' '.join(cmd_list)} in {cwd}", "INFO")
    try:
        process = subprocess.Popen(
            cmd_list, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, universal_newlines=True, errors='replace' # Added errors='replace'
        )

        # Thread to read output without blocking
        def read_output():
            try:
                for line in iter(process.stdout.readline, ''): # type: ignore
                    if line is not None: # Ensure line is not None before processing
                        log_fn(line.rstrip('\n\r')) # Log message without extra newlines
                    if process.poll() is not None: # Process finished
                        break
            except Exception as e:
                log_fn(f"Error reading process output: {e}", "ERROR")
            finally:
                if process.stdout: # type: ignore
                    process.stdout.close() # type: ignore

        output_thread = threading.Thread(target=read_output)
        output_thread.daemon = True # Allow main program to exit even if thread is running
        output_thread.start()

        try:
            process.wait(timeout=timeout) # Wait for the process to complete or timeout
        except subprocess.TimeoutExpired: # This will only be raised if timeout is not None
            log_fn(f"Command timed out after {timeout} seconds: {' '.join(cmd_list)}", "ERROR")
            process.terminate() # Try to terminate gracefully
            try:
                process.wait(timeout=10) # Wait a bit for termination
            except subprocess.TimeoutExpired:
                process.kill() # Force kill if termination fails
                process.wait() # Ensure process is reaped
            raise MergekitError(f"Command timed out: {' '.join(cmd_list)}")

        output_thread.join(timeout=10) # Ensure output thread finishes and logs all output

        if process.returncode != 0:
            log_fn(f"Command failed with exit code {process.returncode}: {' '.join(cmd_list)}", "ERROR")
            raise MergekitError(f"Command failed: {' '.join(cmd_list)}. Exit code: {process.returncode}")

        log_fn(f"Command finished successfully: {' '.join(cmd_list)}", "INFO")

    except Exception as e:
        log_fn(f"Exception during command execution '{' '.join(cmd_list)}': {str(e)}", "ERROR")
        if not isinstance(e, MergekitError): # Don't wrap MergekitError in another MergekitError
            raise MergekitError(f"Execution failed for '{' '.join(cmd_list)}': {str(e)}")
        else:
            raise # Re-raise the original MergekitError

def _prefetch_models_with_logging(
    merge_config: MergeConfiguration, hf_home: str, lora_merge_cache: str, log_fn: Callable,
    hf_token: Optional[str] = None, trust_remote_code_prefetch: bool = False
):
    log_fn(f"Prefetching models. HF_HOME: {hf_home}, Lora Cache: {lora_merge_cache}", "INFO")
    for i, model_source_info in enumerate(merge_config.referenced_models()):
        log_fn(f"Prefetching model {i+1}: {model_source_info.model}", "INFO")
        try:
            start_time = time.time()
            # Get merged model reference (handles LoRAs etc.)
            model_ref = model_source_info.merged(
                cache_dir=lora_merge_cache,
                huggingface_token=hf_token,
                trust_remote_code=trust_remote_code_prefetch
            )
            # Now download/resolve the actual model files for this reference
            log_fn(f"Getting local path for {model_ref.model}", "DEBUG")
            local_model_path = model_ref.local_path(
                cache_dir=hf_home, # This is where the main model weights are stored
                huggingface_token=hf_token,
                trust_remote_code=trust_remote_code_prefetch
            )
            elapsed_time = time.time() - start_time
            log_fn(f"Model {model_source_info.model} prefetched in {elapsed_time:.2f}s -> {local_model_path}", "INFO")
        except Exception as e:
            log_fn(f"Error prefetching model {model_source_info.model}: {str(e)}", "ERROR")
            # Optionally, re-raise or handle to stop the whole process if a prefetch fails
            continue # Continue to prefetch other models

def _run_merge_operation(
    current_has_gpu: bool, actual_cli_for_mergekit: str, merged_path_workdir: str,
    output_subdir_name: str, # This is the 'output' arg to mergekit-yaml (e.g. "merged_model_output")
    operational_temp_dir_str: str, # This is the root of our temp working area (e.g. /tmp/mergekit_op_XXXX)
    log_fn: Callable,
    low_cpu_mem: bool = True, # Defaulted from previous logic
    read_to_gpu: bool = True, # Defaulted from previous logic, seems unused directly
    trust_remote_code_config: bool = False
):
    log_fn("Setting up merge environment", "INFO")
    tmp_env = os.environ.copy()

    # Ensure paths are absolute and correctly used for cache locations
    abs_operational_temp_dir = str(pathlib.Path(operational_temp_dir_str).resolve())
    tmp_env["HF_HOME"] = str(pathlib.Path(abs_operational_temp_dir) / ".cache" / "hf")
    tmp_env["TMPDIR"] = str(pathlib.Path(abs_operational_temp_dir) / ".cache" / "tmp") # For general temp files by underlying libs
    tmp_env["TRANSFORMERS_CACHE"] = str(pathlib.Path(abs_operational_temp_dir) / ".cache" / "transformers")
    tmp_env["HF_HUB_CACHE"] = tmp_env["HF_HOME"] # Often interchangeable with HF_HOME for hub downloads
    tmp_env["PYTHONUNBUFFERED"] = "1" # For immediate log output

    # Create cache directories
    os.makedirs(tmp_env["HF_HOME"], exist_ok=True)
    os.makedirs(tmp_env["TMPDIR"], exist_ok=True)
    os.makedirs(tmp_env["TRANSFORMERS_CACHE"], exist_ok=True)

    # Lora merge cache argument for the CLI
    lora_cache_arg = str(pathlib.Path(abs_operational_temp_dir) / ".cache" / "lora_merge_cache")
    os.makedirs(lora_cache_arg, exist_ok=True)

    # Construct the command list properly using shlex for robustness
    full_cli_list = shlex.split(actual_cli_for_mergekit)

    # Add --lora-merge-cache if not present
    if "--lora-merge-cache" not in full_cli_list:
        full_cli_list.extend(["--lora-merge-cache", lora_cache_arg])

    if trust_remote_code_config and "--trust-remote-code" not in full_cli_list:
        full_cli_list.append("--trust-remote-code")

    if current_has_gpu:
        log_fn(f"Configuring for GPU merge", "INFO")
        if "--cuda" not in full_cli_list:
            full_cli_list.append("--cuda")
        if low_cpu_mem and "--low-cpu-memory" not in full_cli_list: # Recommended for GPU merges
             full_cli_list.append("--low-cpu-memory")
    else: # CPU merge
        log_fn(f"Configuring for CPU merge", "INFO")
        # --lazy-unpickle can be useful for CPU merges with large models
        if "--lazy-unpickle" not in full_cli_list:
            full_cli_list.append("--lazy-unpickle")
            
    log_fn(f"Final merge command list: {full_cli_list}", "DEBUG")

    try:
        _run_command_with_logging(
            full_cli_list,
            cwd=str(merged_path_workdir), # Execute in the directory containing config.yaml
            env=tmp_env,
            log_fn=log_fn,
            timeout=None # Potentially long running, consider a very long timeout or None
        )
    except MergekitError as e:
        log_fn(f"Merge command failed: {str(e)}", "ERROR")
        raise # Re-raise to be caught by the caller
    except Exception as e: # Catch any other unexpected errors
        log_fn(f"Unexpected error during merge operation setup: {str(e)}", "ERROR")
        raise MergekitError(f"Merge operation setup failed: {str(e)}")


def process_model_merge(
    yaml_config_str: str, hf_token_merge: Optional[str], repo_name: Optional[str],
    local_path_merge_output: Optional[str], community_hf_token_val: Optional[str], use_gpu_bool: bool,
    temp_dir_base: str, # Fallback if MERGEKIT_JOB_TEMP_DIR is not set
    log_fn: Callable, trust_remote_code_config: bool = False,
    trust_remote_code_model_ops: bool = False # For model loading/prefetching
) -> Tuple[Optional[Iterable[str]], Optional[str], Optional[str]]:
    log_fn("=== Starting process_model_merge ===", "INFO")
    final_local_path_str: Optional[str] = None
    error_message_str: Optional[str] = None
    uploaded_files_list: Optional[Iterable[str]] = None # Not currently populated, but kept for signature

    try:
        current_has_gpu = torch.cuda.is_available() if use_gpu_bool else False
        log_fn(f"GPU check: Requested GPU={use_gpu_bool}, System has GPU={torch.cuda.is_available()}, Using GPU={current_has_gpu}", "INFO")

        if not yaml_config_str or not yaml_config_str.strip():
            log_fn("Empty YAML configuration provided.", "ERROR")
            return None, None, "Empty YAML configuration."

        log_fn("Validating YAML configuration...", "INFO")
        try:
            parsed_yaml = yaml.safe_load(yaml_config_str)
            merge_config = MergeConfiguration.model_validate(parsed_yaml)
            log_fn("YAML configuration validated successfully.", "INFO")
        except Exception as e:
            log_fn(f"Invalid YAML configuration: {e}", "ERROR")
            return None, None, f"Invalid YAML: {str(e)}"

        # Determine the effective temporary directory base
        env_temp_dir_override = os.environ.get("MERGEKIT_JOB_TEMP_DIR")
        effective_temp_dir_for_ops: str

        if env_temp_dir_override:
            log_fn(f"Using temp directory base from environment variable MERGEKIT_JOB_TEMP_DIR: {env_temp_dir_override}", "INFO")
            effective_temp_dir_for_ops = env_temp_dir_override
            try:
                pathlib.Path(effective_temp_dir_for_ops).mkdir(parents=True, exist_ok=True)
            except Exception as e_mkdir:
                log_fn(f"Failed to create directory from MERGEKIT_JOB_TEMP_DIR ('{effective_temp_dir_for_ops}'): {e_mkdir}. Falling back.", "ERROR")
                # Fallback to temp_dir_base if creation fails
                effective_temp_dir_for_ops = temp_dir_base
                log_fn(f"Fallen back to using temp_dir_base from argument: {effective_temp_dir_for_ops}", "INFO")
                # Ensure fallback directory exists
                pathlib.Path(effective_temp_dir_for_ops).mkdir(parents=True, exist_ok=True)
        else:
            effective_temp_dir_for_ops = temp_dir_base
            log_fn(f"Using temp directory base from function argument: {effective_temp_dir_for_ops}", "INFO")
            # Ensure the argument-provided directory exists (it should be by combined_app.py, but good practice)
            pathlib.Path(effective_temp_dir_for_ops).mkdir(parents=True, exist_ok=True)


        # Determine effective Hugging Face token
        effective_hf_token = hf_token_merge
        if not effective_hf_token and repo_name and repo_name.startswith("mergekit-community/"):
            effective_hf_token = community_hf_token_val
        
        log_fn(f"Creating operational temp directory within: {effective_temp_dir_for_ops}", "INFO")
        with tempfile.TemporaryDirectory(prefix="mergekit_op_", dir=effective_temp_dir_for_ops) as operational_temp_dir_str:
            log_fn(f"Operational temp directory created: {operational_temp_dir_str}", "DEBUG")
            operational_temp_dir = pathlib.Path(operational_temp_dir_str)

            # Directory where config.yaml will be placed and mergekit-yaml will be run from
            merge_process_workdir = operational_temp_dir / "merge_process_work"
            merge_process_workdir.mkdir(parents=True, exist_ok=True)

            config_yaml_path = merge_process_workdir / "config.yaml"
            config_yaml_path.write_text(yaml_config_str)
            log_fn(f"Configuration saved to: {config_yaml_path}", "INFO")

            # Name of the subdirectory mergekit will create *inside* merge_process_workdir
            merge_output_subdir_name = "merged_model_output" 
            # This is the path mergekit-yaml will write to, relative to merge_process_workdir
            mergekit_internal_output_dir = merge_process_workdir / merge_output_subdir_name

            # Construct base mergekit command string
            # mergekit-yaml <config_path> <output_path> [options]
            # Paths are relative to the CWD of the command (merge_process_workdir)
            base_mergekit_command_str = f"mergekit-yaml {config_yaml_path.name} {merge_output_subdir_name} --copy-tokenizer"

            if "--allow-crimes" not in base_mergekit_command_str: # Often needed
                 base_mergekit_command_str += " --allow-crimes"
            
            # Model Prefetching (uses caches within operational_temp_dir)
            log_fn("Starting model prefetch...", "INFO")
            try:
                _prefetch_models_with_logging(
                    merge_config,
                    hf_home=str(operational_temp_dir / ".cache" / "hf"),
                    lora_merge_cache=str(operational_temp_dir / ".cache" / "lora_merge_cache"),
                    log_fn=log_fn,
                    hf_token=effective_hf_token,
                    trust_remote_code_prefetch=trust_remote_code_model_ops
                )
                log_fn("Model prefetch completed.", "INFO")
            except Exception as e:
                log_fn(f"Error during model prefetch: {str(e)}", "WARNING")
                # Depending on severity, might want to `return None, None, str(e)` here

            # Run Merge Operation
            log_fn("Starting merge operation...", "INFO")
            try:
                _run_merge_operation(
                    current_has_gpu, base_mergekit_command_str, str(merge_process_workdir),
                    merge_output_subdir_name, str(operational_temp_dir), log_fn,
                    trust_remote_code_config=trust_remote_code_config
                )
                log_fn("Merge operation completed successfully.", "INFO")
            except MergekitError as e:
                log_fn(f"Merge operation failed: {str(e)}", "ERROR")
                return None, None, f"Merge operation failed: {str(e)}"
            except Exception as e: # Catch any other unexpected errors from _run_merge_operation
                log_fn(f"Unexpected failure during merge operation: {str(e)}", "ERROR")
                return None, None, f"Unexpected merge failure: {str(e)}"

            if not mergekit_internal_output_dir.exists() or not mergekit_internal_output_dir.is_dir():
                msg = f"Mergekit output directory not found after merge: {mergekit_internal_output_dir}"
                log_fn(msg, "ERROR")
                return None, None, msg
            
            log_fn(f"Mergekit output found at: {mergekit_internal_output_dir}", "INFO")

            # Handling final output path (copy from temp mergekit output)
            if local_path_merge_output:
                target_local_dir = pathlib.Path(local_path_merge_output)
                log_fn(f"Copying merged model from {mergekit_internal_output_dir} to final local path: {target_local_dir}", "INFO")
                try:
                    target_local_dir.parent.mkdir(parents=True, exist_ok=True) # Ensure parent of target exists
                    if target_local_dir.exists():
                        log_fn(f"Target local path {target_local_dir} exists, removing before copy.", "WARNING")
                        shutil.rmtree(target_local_dir)
                    shutil.copytree(mergekit_internal_output_dir, target_local_dir, dirs_exist_ok=False) # copy content
                    log_fn(f"Model successfully copied to: {target_local_dir}", "INFO")
                    final_local_path_str = str(target_local_dir)
                except Exception as e:
                    log_fn(f"Error copying to local path {target_local_dir}: {e}", "ERROR")
                    return None, None, f"Error copying to local path: {str(e)}" # Critical error
            else:
                # If no specific local_path_merge_output, the "final" path is the one in the temp operational dir
                # This might be used if the model is only for direct GGUF conversion passthrough
                log_fn("No final local save path specified. Merged model exists in temp.", "INFO")
                final_local_path_str = str(mergekit_internal_output_dir) # Pass this path to GGUF step

            # Hugging Face Upload
            upload_to_hf = bool(repo_name)
            repo_url_str: Optional[str] = None
            actual_repo_name_used_for_upload = repo_name

            if upload_to_hf:
                log_fn(f"Preparing to upload to Hugging Face repo: {repo_name}", "INFO")
                if not effective_hf_token:
                    if repo_name.startswith("mergekit-community/") and community_hf_token_val: # type: ignore
                        effective_hf_token = community_hf_token_val
                        log_fn(f"Using mergekit-community token for upload to {repo_name}.", "INFO")
                    else:
                        msg = f"Cannot upload to '{repo_name}'. No Hugging Face token provided."
                        log_fn(msg, "ERROR")
                        error_message_str = msg # Report error but continue if local save was done
                        upload_to_hf = False # Prevent upload attempt

                if upload_to_hf and effective_hf_token: # Re-check after token logic
                    try:
                        hf_api_for_upload = huggingface_hub.HfApi(token=effective_hf_token)
                        # Ensure repo_name includes namespace if not provided
                        if actual_repo_name_used_for_upload and '/' not in actual_repo_name_used_for_upload:
                            try:
                                user_info = hf_api_for_upload.whoami() # Requires token with read access
                                user_namespace = user_info['name']
                                actual_repo_name_used_for_upload = f"{user_namespace}/{actual_repo_name_used_for_upload}"
                                log_fn(f"Repo name updated with user namespace: {actual_repo_name_used_for_upload}", "INFO")
                            except Exception as e_whoami:
                                log_fn(f"Could not determine user namespace for repo name ('{actual_repo_name_used_for_upload}'): {e_whoami}. Upload may fail or use unexpected namespace.", "WARNING")
                        
                        repo_url_obj = hf_api_for_upload.create_repo(
                            repo_id=actual_repo_name_used_for_upload, # type: ignore
                            exist_ok=True,
                            private=False # Assuming public for mergekit-community or general merges, could be a param
                        )
                        repo_url_str = repo_url_obj.repo_url if hasattr(repo_url_obj, 'repo_url') else str(repo_url_obj)
                        log_fn(f"HF repo ready for upload: {repo_url_str}", "INFO")

                        log_fn(f"Uploading model from {mergekit_internal_output_dir} to {actual_repo_name_used_for_upload}...", "INFO")
                        # Upload the content of mergekit_internal_output_dir
                        hf_api_for_upload.upload_folder(
                            repo_id=actual_repo_name_used_for_upload, # type: ignore
                            folder_path=str(mergekit_internal_output_dir),
                            commit_message=f"Add merged model: {yaml_config_str[:100]}..." # Brief commit message
                        )
                        log_fn(f"Successfully uploaded to HF repo: {repo_url_str}", "INFO")
                        # uploaded_files_list could be populated here if needed, e.g. by listing files in repo_url_str

                    except Exception as e_hf_upload:
                        upload_err_msg = f"Error during Hugging Face upload to {actual_repo_name_used_for_upload}: {str(e_hf_upload)}"
                        log_fn(upload_err_msg, "ERROR")
                        # Append to existing error_message_str if any, or set it
                        error_message_str = f"{error_message_str}\nAlso, HF Upload Error: {upload_err_msg}" if error_message_str else upload_err_msg

        # End of `with tempfile.TemporaryDirectory` block, operational_temp_dir_str is cleaned up
        log_fn("=== process_model_merge completed ===", "INFO")
        return uploaded_files_list, final_local_path_str, error_message_str

    except Exception as e_global: # Catch-all for unexpected issues in the main try block
        log_fn(f"=== process_model_merge failed with unhandled exception: {str(e_global)} ===", "ERROR")
        import traceback
        log_fn(f"Traceback: {traceback.format_exc()}", "ERROR")
        return None, None, f"Unexpected error in process_model_merge: {str(e_global)}"

if __name__ == "__main__":
    # Example usage for testing (not part of the main application flow)
    print("mergekit_utils.py loaded. Standalone test block.")

    # To test the environment variable functionality:
    # 1. Set MERGEKIT_JOB_TEMP_DIR: export MERGEKIT_JOB_TEMP_DIR=/tmp/my_custom_merge_temp
    # 2. Run this script: python mergekit_utils.py
    # 3. Check if /tmp/my_custom_merge_temp/mergekit_op_* directories are created and used.

    # Mock log_fn for testing
    def test_logger(message, level="INFO"):
        print(f"[{level}] {message}")

    # Example YAML (replace with a valid simple one for testing if needed)
    test_yaml = """
models:
  - model: cognitivecomputations/dolphin-2.6-mistral-7b-dpo-laser 
    # Using a small, fast-downloading model might be better for quick tests if available
    # For actual merge, ensure this model is accessible
    parameters:
      weight: 1.0
merge_method: passthrough
dtype: float16
"""
    # Create a dummy temp_dir_base for fallback testing
    fallback_temp_base = pathlib.Path("./test_fallback_temp_base")
    fallback_temp_base.mkdir(parents=True, exist_ok=True)

    print(f"Testing process_model_merge. Fallback base: {fallback_temp_base.resolve()}")
    # Minimal call to see if temp dir logic works (will likely fail on actual merge without setup)
    try:
        _, out_path, err = process_model_merge(
            yaml_config_str=test_yaml,
            hf_token_merge=None,
            repo_name=None,
            local_path_merge_output=str(fallback_temp_base / "test_output_model"),
            community_hf_token_val=None,
            use_gpu_bool=False,
            temp_dir_base=str(fallback_temp_base.resolve()),
            log_fn=test_logger
        )
        if err:
            print(f"Test run resulted in error: {err}")
        if out_path:
            print(f"Test run output path: {out_path}")
        if not err and not out_path:
             print("Test run completed, but no output path or error reported (may indicate early exit or prefetch/merge tool failure).")

    except Exception as e:
        print(f"Test run threw exception: {e}")
    finally:
        # Clean up dummy fallback base if it was used
        # shutil.rmtree(fallback_temp_base, ignore_errors=True)
        print(f"Test cleanup: If you used MERGEKIT_JOB_TEMP_DIR, check that directory. Fallback dir was {fallback_temp_base.resolve()}")
        print("If merge ran, temp mergekit_op_* folders should be inside the effective_temp_dir_for_ops.")