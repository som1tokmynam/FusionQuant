import os
from pathlib import Path # Corrected import
import random
import string
import tempfile
import time
import shutil
import subprocess
import threading
import signal
from typing import Iterable, List, Tuple, Optional, Callable, Any, Union

import huggingface_hub
import torch # Used to check torch.cuda.is_available()
import yaml
from mergekit.config import MergeConfiguration # Assuming this is how it's imported
import shlex

class MergekitError(Exception):
    """Custom exception for Mergekit errors."""
    pass

def get_example_yaml_filenames_for_gr_examples(examples_dir_str: str = "examples") -> List[str]:
    """Get list of example YAML filenames for Gradio examples."""
    examples_path = Path(examples_dir_str) # Now Path is defined
    filenames = []

    if examples_path.exists() and examples_path.is_dir():
        for yaml_file in examples_path.glob("*.yaml"):
            try:
                filenames.append(yaml_file.name)
            except Exception as e:
                print(f"Error accessing example file {yaml_file}: {e}")

    if not filenames: # If no files found, try to create/ensure default
        if not (examples_path / "default_example.yaml").exists():
            try:
                create_example_files(examples_dir_str) # Attempt to create them
                if (examples_path / "default_example.yaml").exists(): # Check again
                    filenames.append("default_example.yaml")
            except Exception as e:
                print(f"Failed to create default example in get_example_yaml_filenames_for_gr_examples: {e}")
        elif (examples_path / "default_example.yaml").exists(): # If it existed but was the only one
             filenames.append("default_example.yaml")

    if not filenames: # Absolute fallback
        filenames.append("default_example.yaml")
    return filenames

def get_example_yaml_content(filename: str, examples_dir_str: str = "examples") -> str:
    """Get the content of a specific example YAML file."""
    examples_path = Path(examples_dir_str) # Now Path is defined
    yaml_file_path = examples_path / filename

    if filename == "default_example.yaml" and not yaml_file_path.exists(): # Special handling for default if missing
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
        if filename != "default_example.yaml":
            return get_example_yaml_content("default_example.yaml", examples_dir_str)
        return f"# Error reading {filename}\n# {str(e)}"


def create_example_files(examples_dir_str: str = "examples"):
    """Create example YAML files if they don't exist."""
    examples_path = Path(examples_dir_str) # Now Path is defined
    examples_path.mkdir(exist_ok=True)

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
        temp_dir_root = Path(tempfile.gettempdir()) # Now Path is defined
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
                elif item.is_file() and (pattern.endswith(".cache") or pattern.endswith(".lora_cache")):
                    try:
                        print(f"Cleaning temporary file: {item}")
                        item.unlink()
                        cleaned_count +=1
                    except Exception as e:
                        print(f"Failed to remove {item}: {e}")
    except Exception as e:
        print(f"Error during cleanup of temp folders: {e}")
    return cleaned_count

def _run_command_with_logging(
    cmd_list: List[str],
    cwd: Union[str, Path], # Path is now defined
    env: Optional[dict],
    log_fn: Callable[[str, str], None],
    timeout: Optional[int] = 3600
):
    log_fn(f"Running command: {' '.join(cmd_list)} in {cwd}", "INFO")
    try:
        process = subprocess.Popen(
            cmd_list, cwd=str(cwd), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, universal_newlines=True, errors='replace'
        )

        def read_output():
            try:
                for line in iter(process.stdout.readline, ''):
                    if line is not None:
                        log_fn(line.rstrip('\n\r'), "INFO")
                    if process.poll() is not None:
                        break
            except Exception as e:
                log_fn(f"Error reading process output: {e}", "ERROR")
            finally:
                if process.stdout:
                    process.stdout.close()

        output_thread = threading.Thread(target=read_output)
        output_thread.daemon = True
        output_thread.start()

        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            log_fn(f"Command timed out after {timeout} seconds: {' '.join(cmd_list)}", "ERROR")
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise MergekitError(f"Command timed out: {' '.join(cmd_list)}")

        output_thread.join(timeout=10)

        if process.returncode != 0:
            log_fn(f"Command failed with exit code {process.returncode}: {' '.join(cmd_list)}", "ERROR")
            raise MergekitError(f"Command failed: {' '.join(cmd_list)}. Exit code: {process.returncode}")

        log_fn(f"Command finished successfully: {' '.join(cmd_list)}", "INFO")

    except Exception as e:
        log_fn(f"Exception during command execution '{' '.join(cmd_list)}': {str(e)}", "ERROR")
        if not isinstance(e, MergekitError):
            raise MergekitError(f"Execution failed for '{' '.join(cmd_list)}': {str(e)}")
        else:
            raise

def _prefetch_models_with_logging(
    merge_config: MergeConfiguration,
    hf_home: str,
    lora_merge_cache: str,
    log_fn: Callable[[str,str], None],
    #hf_token: Optional[str] = None,
    #trust_remote_code_prefetch: bool = False
):
    log_fn(f"Prefetching models. Using HF Cache: {hf_home}, Lora Cache: {lora_merge_cache}", "INFO")
    Path(hf_home).mkdir(parents=True, exist_ok=True) # Path is defined
    Path(lora_merge_cache).mkdir(parents=True, exist_ok=True) # Path is defined

    for i, model_source_info in enumerate(merge_config.referenced_models()):
        log_fn(f"Prefetching model {i+1}: {model_source_info.model}", "INFO")
        try:
            start_time = time.time()
            model_ref = model_source_info.merged(
                cache_dir=lora_merge_cache,
                # huggingface_token=hf_token, not working
                #trust_remote_code=trust_remote_code_prefetch
            )
            log_fn(f"Getting local path for {model_ref.model} using main HF cache: {hf_home}", "DEBUG")
            local_model_path = model_ref.local_path(
                cache_dir=hf_home,
                # huggingface_token=hf_token, not working
                #trust_remote_code=trust_remote_code_prefetch
            )
            elapsed_time = time.time() - start_time
            log_fn(f"Model {model_source_info.model} prefetched in {elapsed_time:.2f}s -> {local_model_path}", "INFO")
        except Exception as e:
            log_fn(f"Error prefetching model {model_source_info.model}: {str(e)}", "ERROR")
            continue

def _run_merge_operation(
    current_has_gpu: bool,
    actual_cli_for_mergekit: str,
    merged_path_workdir: str,
    output_subdir_name: str,
    operational_temp_dir_str: str,
    hf_home_for_this_op: str,
    log_fn: Callable[[str,str], None],
    low_cpu_mem: bool = True,
    trust_remote_code_config: bool = False
):
    log_fn("Setting up merge environment", "INFO")
    tmp_env = os.environ.copy()
    abs_operational_temp_dir = str(Path(operational_temp_dir_str).resolve()) # Path is defined

    tmp_env["HF_HOME"] = hf_home_for_this_op
    tmp_env["HF_HUB_CACHE"] = hf_home_for_this_op

    tmp_env["TMPDIR"] = str(Path(abs_operational_temp_dir) / ".cache" / "tmp") # Path is defined
    tmp_env["TRANSFORMERS_CACHE"] = str(Path(abs_operational_temp_dir) / ".cache" / "transformers") # Path is defined
    tmp_env["PYTHONUNBUFFERED"] = "1"

    os.makedirs(tmp_env["HF_HOME"], exist_ok=True)
    os.makedirs(tmp_env["TMPDIR"], exist_ok=True)
    os.makedirs(tmp_env["TRANSFORMERS_CACHE"], exist_ok=True)

    lora_cache_arg_path = str(Path(abs_operational_temp_dir) / ".cache" / "lora_merge_cache") # Path is defined
    os.makedirs(lora_cache_arg_path, exist_ok=True)

    full_cli_list = shlex.split(actual_cli_for_mergekit)
    if "--lora-merge-cache" not in full_cli_list:
        full_cli_list.extend(["--lora-merge-cache", lora_cache_arg_path])
    if trust_remote_code_config and "--trust-remote-code" not in full_cli_list:
        full_cli_list.append("--trust-remote-code")
    if current_has_gpu:
        log_fn(f"Configuring for GPU merge", "INFO")
        if "--cuda" not in full_cli_list: full_cli_list.append("--cuda")
        if low_cpu_mem and "--low-cpu-memory" not in full_cli_list: full_cli_list.append("--low-cpu-memory")
    else:
        log_fn(f"Configuring for CPU merge", "INFO")
        if "--lazy-unpickle" not in full_cli_list: full_cli_list.append("--lazy-unpickle")

    log_fn(f"Final merge command list: {full_cli_list}", "DEBUG")
    try:
        _run_command_with_logging(full_cli_list, cwd=str(merged_path_workdir), env=tmp_env, log_fn=log_fn, timeout=None)
    except MergekitError as e:
        log_fn(f"Merge command failed: {str(e)}", "ERROR"); raise
    except Exception as e:
        log_fn(f"Unexpected error during merge operation setup: {str(e)}", "ERROR")
        raise MergekitError(f"Merge operation setup failed: {str(e)}")


def process_model_merge(
    yaml_config_str: str,
    hf_token_merge: Optional[str],
    repo_name: Optional[str],
    local_path_merge_output: Optional[str],
    community_hf_token_val: Optional[str],
    use_gpu_bool: bool,
    temp_dir_base: str,
    log_fn: Callable[[str,str], None],
    # MODIFIED: Moved hf_repo_private_for_merge to be with other default arguments
    hf_repo_private_for_merge: bool = False,
    trust_remote_code_config: bool = False,
    trust_remote_code_model_ops: bool = False,
    keep_hf_cache: bool = False,
    persistent_hf_cache_path: Optional[str] = None
) -> Tuple[Optional[Iterable[str]], Optional[str], Optional[str]]:
    log_fn("=== Starting process_model_merge ===", "INFO")
    error_message_str: Optional[str] = None
    uploaded_files_list: Optional[Iterable[str]] = None

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

        env_temp_dir_override = os.environ.get("MERGEKIT_JOB_TEMP_DIR")
        effective_temp_dir_for_ops: str
        if env_temp_dir_override:
            effective_temp_dir_for_ops = env_temp_dir_override
            try: Path(effective_temp_dir_for_ops).mkdir(parents=True, exist_ok=True) # Path is defined
            except Exception as e_mkdir:
                log_fn(f"Failed to create dir from MERGEKIT_JOB_TEMP_DIR ('{effective_temp_dir_for_ops}'): {e_mkdir}. Falling back.", "ERROR")
                effective_temp_dir_for_ops = temp_dir_base
                log_fn(f"Fallen back to using temp_dir_base from argument: {effective_temp_dir_for_ops}", "INFO")
                Path(effective_temp_dir_for_ops).mkdir(parents=True, exist_ok=True) # Path is defined
        else:
            effective_temp_dir_for_ops = temp_dir_base
            log_fn(f"Using temp directory base from function argument: {effective_temp_dir_for_ops}", "INFO")
            Path(effective_temp_dir_for_ops).mkdir(parents=True, exist_ok=True) # Path is defined

        effective_hf_token = hf_token_merge

        with tempfile.TemporaryDirectory(prefix="mergekit_op_", dir=effective_temp_dir_for_ops) as operational_temp_dir_str:
            operational_temp_dir = Path(operational_temp_dir_str) # Path is defined
            log_fn(f"Operational temp directory for non-HF cache items: {operational_temp_dir}", "DEBUG")

            hf_home_for_operation: str
            if keep_hf_cache and persistent_hf_cache_path and persistent_hf_cache_path.strip():
                persistent_cache_path_obj = Path(persistent_hf_cache_path.strip()).resolve() # Path is defined
                persistent_cache_path_obj.mkdir(parents=True, exist_ok=True)
                hf_home_for_operation = str(persistent_cache_path_obj)
                log_fn(f"Using persistent Hugging Face cache for merge operations: {hf_home_for_operation}", "INFO")
            else:
                temp_hf_home_path = operational_temp_dir / ".cache" / "hf_temp_merge_cache"
                hf_home_for_operation = str(temp_hf_home_path)
                log_fn(f"Using temporary Hugging Face cache for merge (will be deleted with op temp dir): {hf_home_for_operation}", "INFO")

            # Determine the working directory and CLI output path based on whether a local save path is provided.
            merge_process_workdir: Path
            output_path_for_cli: str
            final_local_path_str: Optional[str]
            is_permanent_local_save = bool(local_path_merge_output and local_path_merge_output.strip())

            if is_permanent_local_save:
                # Use the user's specified path as the work directory. This avoids a final copy.
                merge_process_workdir = Path(local_path_merge_output)
                # Ensure the directory is clean before merging into it.
                if merge_process_workdir.exists():
                    log_fn(f"Target local path {merge_process_workdir} exists, removing before merge.", "WARNING")
                    shutil.rmtree(merge_process_workdir)
                # The CLI will output to its current working directory.
                output_path_for_cli = "."
                final_local_path_str = str(merge_process_workdir)
            else:
                # No local path specified, so work inside a temporary subdirectory.
                merge_process_workdir = operational_temp_dir / "merge_process_work"
                # The CLI will output to a named subdirectory within the temp work dir.
                output_path_for_cli = "merged_model_output"
                # The final path will point to this temporary location.
                final_local_path_str = str(merge_process_workdir / output_path_for_cli)

            merge_process_workdir.mkdir(parents=True, exist_ok=True)
            config_yaml_path = merge_process_workdir / "config.yaml"
            config_yaml_path.write_text(yaml_config_str)
            log_fn(f"Configuration saved to: {config_yaml_path}", "INFO")

            # Use the dynamically determined output path for the mergekit command.
            base_mergekit_command_str = f"mergekit-yaml {config_yaml_path.name} {output_path_for_cli} --copy-tokenizer"

            if "--allow-crimes" not in base_mergekit_command_str: base_mergekit_command_str += " --allow-crimes"

            log_fn("Starting model prefetch...", "INFO")
            try:
                _prefetch_models_with_logging(
                    merge_config,
                    hf_home=hf_home_for_operation,
                    lora_merge_cache=str(operational_temp_dir / ".cache" / "lora_merge_cache"),
                    log_fn=log_fn,
                    # hf_token=effective_hf_token,
                    # trust_remote_code_prefetch=trust_remote_code_model_ops
                )
                log_fn("Model prefetch completed.", "INFO")
            except Exception as e_prefetch:
                log_fn(f"Error during model prefetch: {str(e_prefetch)}", "WARNING")

            log_fn("Starting merge operation...", "INFO")
            try:
                _run_merge_operation(
                    current_has_gpu, base_mergekit_command_str, str(merge_process_workdir),
                    output_path_for_cli, # Use dynamic output path name for logging
                    str(operational_temp_dir),
                    hf_home_for_this_op=hf_home_for_operation,
                    log_fn=log_fn,
                    trust_remote_code_config=trust_remote_code_config
                )
                log_fn("Merge operation completed successfully.", "INFO")
            except MergekitError as e_merge_op:
                log_fn(f"Merge operation failed: {str(e_merge_op)}", "ERROR")
                return None, None, f"Merge operation failed: {str(e_merge_op)}"
            except Exception as e_merge_unexpected:
                log_fn(f"Unexpected failure during merge operation: {str(e_merge_unexpected)}", "ERROR")
                return None, None, f"Unexpected merge failure: {str(e_merge_unexpected)}"

            # The output directory is now the final local path itself.
            output_dir_after_merge = Path(final_local_path_str)
            if not output_dir_after_merge.exists() or not output_dir_after_merge.is_dir():
                msg = f"Mergekit output directory not found after merge: {output_dir_after_merge}"
                log_fn(msg, "ERROR"); return None, None, msg
            log_fn(f"Mergekit output found at: {output_dir_after_merge}", "INFO")

            # The copy operation is no longer needed as we merged directly into the final directory.
            # The 'final_local_path_str' is already set correctly above.
            log_fn(f"Merged model is at its final destination: {final_local_path_str}", "INFO")

            upload_to_hf = bool(repo_name)
            actual_repo_name_used_for_upload = repo_name
            if upload_to_hf:
                log_fn(f"Preparing to upload to Hugging Face repo: {repo_name}", "INFO")
                if not effective_hf_token:
                    if repo_name and repo_name.startswith("mergekit-community/") and community_hf_token_val:
                        effective_hf_token = community_hf_token_val
                        log_fn(f"Using mergekit-community token for upload to {repo_name}.", "INFO")
                    else:
                        msg = f"Cannot upload to '{repo_name}'. No Hugging Face token provided."
                        log_fn(msg, "ERROR"); error_message_str = msg; upload_to_hf = False

                if upload_to_hf and effective_hf_token:
                    try:
                        hf_api_for_upload = huggingface_hub.HfApi(token=effective_hf_token)
                        if actual_repo_name_used_for_upload and '/' not in actual_repo_name_used_for_upload:
                            try:
                                user_info = hf_api_for_upload.whoami(); user_namespace = user_info['name']
                                actual_repo_name_used_for_upload = f"{user_namespace}/{actual_repo_name_used_for_upload}"
                                log_fn(f"Repo name updated with user namespace: {actual_repo_name_used_for_upload}", "INFO")
                            except Exception as e_whoami:
                                log_fn(f"Could not determine user namespace for repo name: {e_whoami}.", "WARNING")

                        repo_url_obj = hf_api_for_upload.create_repo(
                            repo_id=actual_repo_name_used_for_upload, # type: ignore
                            exist_ok=True,
                            private=hf_repo_private_for_merge
                        )
                        repo_url_str = str(repo_url_obj) # Get string representation
                        log_fn(f"HF repo ready for upload: {repo_url_str}", "INFO")

                        log_fn(f"Uploading model from {output_dir_after_merge} to {actual_repo_name_used_for_upload}...", "INFO")
                        hf_api_for_upload.upload_folder(
                            repo_id=actual_repo_name_used_for_upload, # type: ignore
                            folder_path=str(output_dir_after_merge),
                            commit_message=f"Add merged model via mergekit: {Path(config_yaml_path).name}" # Path is defined
                        )
                        log_fn(f"Successfully uploaded to HF repo: {repo_url_str}", "INFO")
                    except Exception as e_hf_upload:
                        upload_err_msg = f"Error during Hugging Face upload to {actual_repo_name_used_for_upload}: {str(e_hf_upload)}"
                        log_fn(upload_err_msg, "ERROR")
                        error_message_str = f"{error_message_str or ''}\nHF Upload Error: {upload_err_msg}".strip()

        log_fn("=== process_model_merge completed ===", "INFO")
        return uploaded_files_list, final_local_path_str, error_message_str

    except Exception as e_global:
        log_fn(f"=== process_model_merge failed with unhandled exception: {str(e_global)} ===", "ERROR")
        import traceback
        log_fn(f"Traceback: {traceback.format_exc()}", "ERROR")
        return None, None, f"Unexpected error in process_model_merge: {str(e_global)}"

if __name__ == "__main__":
    print("mergekit_utils.py loaded. Standalone test block.")
    def test_logger(message, level="INFO"): print(f"[{level}] {message}")
    test_yaml = """
models:
  - model: cognitivecomputations/dolphin-2.6-mistral-7b-dpo-laser
    parameters: {weight: 1.0}
merge_method: passthrough
dtype: float16"""
    fallback_temp_base = Path("./test_fallback_temp_base") # Path is defined
    fallback_temp_base.mkdir(parents=True, exist_ok=True)
    print(f"Testing process_model_merge. Fallback base: {fallback_temp_base.resolve()}")
    try:
        _, out_path, err = process_model_merge(
            yaml_config_str=test_yaml,
            hf_token_merge=None,
            repo_name=None,
            local_path_merge_output=str(fallback_temp_base / "test_output_model"),
            community_hf_token_val=None,
            use_gpu_bool=False,
            temp_dir_base=str(fallback_temp_base.resolve()),
            log_fn=test_logger,
            hf_repo_private_for_merge=False, # Testing the new param in correct order
            keep_hf_cache=True,
            persistent_hf_cache_path=str(fallback_temp_base / "persistent_hf_cache_test")
        )
        if err: print(f"Test run resulted in error: {err}")
        if out_path: print(f"Test run output path: {out_path}")
        if not err and not out_path: print("Test run completed, but no output path or error reported.")
    except Exception as e: print(f"Test run threw exception: {e}")
    finally:
        print(f"Test cleanup: Check {fallback_temp_base.resolve()} for outputs and persistent cache.")