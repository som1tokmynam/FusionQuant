import os
import gradio as gr
import tempfile
import shutil
from pathlib import Path
import queue
from threading import Thread
import datetime

# --- Utility Modules ---
try:
    import gguf_utils
    if not hasattr(gguf_utils, 'GGUFConversionError'):
        class GGUFConversionError(Exception): pass
        if gguf_utils: gguf_utils.GGUFConversionError = GGUFConversionError
except ImportError:
    print("WARNING: gguf_utils.py not found. GGUF conversion will not work.")
    gguf_utils = None
    class GGUFConversionError(Exception): pass

try:
    import mergekit_utils
    from gradio_logsview.logsview import LogsView, Log # type: ignore
except ImportError as e:
    print(f"WARNING: mergekit_utils.py or gradio_logsview not found. Model merging/logging will be basic. Error: {e}")
    mergekit_utils = None
    if 'LogsView' not in globals(): LogsView = gr.Textbox # type: ignore
    if 'Log' not in globals():
        class Log: # type: ignore
            def __init__(self, message, level, timestamp):
                self.message, self.level, self.timestamp = message, level, timestamp

# --- Environment & Constants ---
HF_TOKEN = os.environ.get("HF_TOKEN") # Global fallback

# MODIFICATION START: Read APP_TEMP_ROOT environment variable for TEMP_DIR_ROOT
APP_TEMP_ROOT_ENV_VAR = os.environ.get("APP_TEMP_ROOT")
if APP_TEMP_ROOT_ENV_VAR:
    TEMP_DIR_ROOT = Path(APP_TEMP_ROOT_ENV_VAR)
    print(f"INFO: Using temporary root directory from APP_TEMP_ROOT environment variable: {TEMP_DIR_ROOT}")
else:
    TEMP_DIR_ROOT = Path("outputs/combined_app_temp") # Default value
    print(f"INFO: APP_TEMP_ROOT environment variable not set. Using default temporary root: {TEMP_DIR_ROOT}")
# MODIFICATION END

TEMP_DIR_ROOT.mkdir(parents=True, exist_ok=True)
BUNDLED_IMATRIX_PATH = "/home/user/app/groups_merged.txt" # Defined constant for the path

# --- Helper Functions (Defined before UI that might use them in .load) ---
def get_hf_token_status(token, token_name):
    return f"✅ {token_name} found" if token else f"❌ {token_name} not found - set for uploads."

def extract_example_label(_, index, filename=""):
    if not filename: return f"Example {index + 1}"
    name_part = filename.removesuffix(".yaml").removesuffix(".yml").replace("_", " ").replace("-", " ")
    return ' '.join(word.capitalize() for word in name_part.split()) or f"Example {index + 1}"

def load_merge_examples(): # Depends on extract_example_label and mergekit_utils
    if not mergekit_utils:
        default_content = "models:\n  - model: cognitivecomputations/dolphin-2.6-mistral-7b-dpo-laser\n    parameters:\n      weight: 0.5\n  - model: OpenPipe/mistral-ft-optimized-1218\n    parameters:\n      weight: 0.5\nmerge_method: linear\ndtype: float16"
        return [[extract_example_label(None, 0, "Default Example.yaml"), default_content]]
    try:
        example_filenames = mergekit_utils.get_example_yaml_filenames_for_gr_examples() # [cite: 4]
        labeled_examples = []
        if not example_filenames or (len(example_filenames) == 1 and "default_example.yaml" in example_filenames[0]):
            default_content = mergekit_utils.get_example_yaml_content("default_example.yaml") # [cite: 4]
            labeled_examples.append([extract_example_label(None, 0, "Default Example.yaml"), default_content])
        else:
            for i, filename in enumerate(example_filenames):
                if filename == "default_example.yaml" and len(example_filenames) > 1:
                    # Skip default if other examples are present, unless it's the *only* one (covered above)
                    # or if somehow it's the only one remaining after filtering (unlikely path here)
                    if not any(f != "default_example.yaml" for f in example_filenames):
                        pass # It's the only one, let it be processed
                    else: # Other examples exist, skip the default_example.yaml in this loop
                        continue
                content = mergekit_utils.get_example_yaml_content(filename) # [cite: 4]
                if content.startswith("# Error reading"): continue # Skip problematic examples
                labeled_examples.append([extract_example_label(None, i, filename), content])
        
        if not labeled_examples: # Fallback if all examples failed to load but default exists or should be used
            default_content = mergekit_utils.get_example_yaml_content("default_example.yaml") # [cite: 4]
            labeled_examples.append([extract_example_label(None, 0, "Default Example.yaml"), default_content])
        return labeled_examples
    except Exception as e:
        print(f"Error loading merge examples: {e}")
        # Fallback to a very basic default if all else fails
        default_content = "models:\n  - model: cognitivecomputations/dolphin-2.6-mistral-7b-dpo-laser\n    parameters:\n      weight: 0.5\n  - model: OpenPipe/mistral-ft-optimized-1218\n    parameters:\n      weight: 0.5\nmerge_method: linear\ndtype: float16"
        return [[extract_example_label(None, 0, "Fallback Linear Merge.yaml"), default_content]]

# --- Gradio Callbacks for Initial UI Population (Defined before .load calls) ---
def populate_examples_for_dropdown():
    examples = load_merge_examples()
    choices = [example[0] for example in examples]
    return gr.update(choices=choices, value=choices[0] if choices else None)

def load_selected_example_content(selected_label):
    if not selected_label: return ""
    examples = load_merge_examples() # Reload examples to ensure consistency
    for label, content in examples:
        if label == selected_label:
            return content
    print(f"Warning: Could not find content for selected label: {selected_label}")
    return "" # Fallback


# --- Gradio UI Definition ---
css = ".gradio-container {overflow-y: auto;} .checkbox-group {max-height: 200px; overflow-y: auto; border: 1px solid #e0e0e0; padding: 10px; margin: 10px 0;} .logs_view_container textarea, .Textbox textarea { font-family: monospace; font-size: 0.85em !important; white-space: pre-wrap !important; }"
with gr.Blocks(css=css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("# FusionQuant Model Merge & GGUF Conversion 🚀")
    merged_model_path_state = gr.State(None)
    gr.Markdown(f"**Global HF_TOKEN Status (Fallback for GGUF & Merge):** {get_hf_token_status(HF_TOKEN, 'HF_TOKEN')}")

    with gr.Tabs():
        with gr.TabItem("Step 1: Merge Models (Mergekit)"):
            gr.Markdown("## Configure and Run Model Merge")
            if not mergekit_utils: gr.Markdown("### ❌ `mergekit_utils.py` not loaded. Merging disabled.")
            with gr.Row():
                with gr.Column(scale=2):
                    merge_yaml_config = gr.Code(label="Mergekit YAML", language="yaml", lines=15)
                    # Ensure example files are created if mergekit_utils is available
                    if mergekit_utils and hasattr(mergekit_utils, 'create_example_files'):
                        try:
                            mergekit_utils.create_example_files() # [cite: 4]
                        except Exception as e:
                            print(f"Error creating merge examples: {e}")
                    
                    gr.Markdown("### Load Merge Example")
                    with gr.Row():
                        example_dropdown = gr.Dropdown(choices=[], label="Select Example", interactive=True)
                        load_example_btn = gr.Button("Load Example", size="sm")
                with gr.Column(scale=1):
                    merge_hf_token_input = gr.Textbox(label="HF Write Token (Mergekit upload)", type="password", placeholder="Uses HF_TOKEN env var if blank.")
                    merge_repo_name = gr.Textbox(label="HF Repo Name (Mergekit upload)", placeholder="e.g., YourUser/MyMerge")
                    merge_local_save_path = gr.Textbox(label="Local Save Path (Merged Model)", placeholder=f"e.g., {TEMP_DIR_ROOT}/my-merged-model") # Uses updated TEMP_DIR_ROOT
                    merge_use_gpu = gr.Checkbox(label="Use GPU for Merge (if available)", value=True, info="Uncheck to force CPU merge.")
                    merge_use_for_gguf = gr.Checkbox(label="✅ Use merged model for GGUF (Step 2)", value=True)

            merge_button = gr.Button("Run Merge", variant="primary", interactive=mergekit_utils is not None)
            merge_status_output = gr.Markdown()
            log_elem_class = "logs_view_container" if LogsView != gr.Textbox else "Textbox"
            if 'LogsView' in globals() and LogsView != gr.Textbox:
                merge_logs_output = LogsView(label="Merge Logs", lines=15, elem_classes=log_elem_class)
            else:
                merge_logs_output = gr.Textbox(label="Merge Logs", lines=15, interactive=False, elem_classes=log_elem_class)

        with gr.TabItem("Step 2: Convert to GGUF & Quantize (Llama.cpp)"):
            gr.Markdown("## Configure and Run GGUF Conversion")
            if not gguf_utils: gr.Markdown("### ❌ `gguf_utils.py` not loaded. GGUF disabled.")
            gguf_model_source = gr.Radio(["Output from Merge Step", "HF Hub", "Local Path"], label="GGUF Model Source", value="Output from Merge Step")
            with gr.Group(visible=False) as gguf_hf_group:
                gguf_model_id = gr.Textbox(label="HF Model ID", placeholder="e.g., meta-llama/Llama-2-7b-hf")
            gguf_local_model_path = gr.Textbox(label="Local Model Path", placeholder="/path/to/model_dir", visible=False)
            gguf_merged_model_display = gr.Textbox(label="Using Merged Model Path:", interactive=False, visible=True)

            gr.Markdown("### Hugging Face Settings (GGUF)")
            gguf_hf_token_input = gr.Textbox(label="HF Token (GGUF Operations)", type="password", placeholder="Uses global HF_TOKEN env var if blank.")

            gr.Markdown("### Quantization Settings")
            gguf_q_methods = gr.CheckboxGroup(["Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q4_0", "Q4_K_S", "Q4_K_M", "Q5_0", "Q5_K_S", "Q5_K_M", "Q6_K", "Q8_0"], label="Standard Quants", value=["Q4_K_M"], elem_classes="checkbox-group")
            gguf_use_imatrix = gr.Checkbox(label="Use Importance Matrix", value=False)
            gguf_imatrix_q_methods = gr.CheckboxGroup(["IQ3_M", "IQ3_XXS", "Q4_K_M", "Q4_K_S", "IQ4_NL", "IQ4_XS", "Q5_K_M", "Q5_K_S"], label="Imatrix Quants", value=["IQ4_NL"], visible=False, elem_classes="checkbox-group")
            
            gguf_use_bundled_imatrix_checkbox = gr.Checkbox(
                label=f"Use bundled groups_merged.txt for Imatrix (Path: {BUNDLED_IMATRIX_PATH})", # [cite: 1]
                value=False, 
                visible=False, 
                info=f"If checked and 'Use Importance Matrix' is active, {BUNDLED_IMATRIX_PATH} will be used." # [cite: 1]
            )

            gr.Markdown("### Output Settings")
            gguf_custom_name_input = gr.Textbox(label="Custom GGUF Base Name (Optional)", placeholder="e.g., MyAwesomeModel-7B")
            gguf_upload_to_hf = gr.Checkbox(label="Upload GGUF to HF", value=True)
            gguf_private_repo = gr.Checkbox(label="Make GGUF Repo Private", value=False, visible=True)
            gguf_local_output_path = gr.Textbox(label="Local Save Path (GGUFs)", placeholder=f"e.g., {TEMP_DIR_ROOT}/gguf_exports") # Uses updated TEMP_DIR_ROOT
            gguf_split_model = gr.Checkbox(label="Split GGUF Model Shards")
            gguf_split_max_tensors = gr.Number(label="Max Tensors/Shard", value=256, visible=False)
            gguf_split_max_size = gr.Textbox(label="Max Size/Shard (e.g., 5G)", visible=False)

            gguf_convert_btn = gr.Button("Convert to GGUF & Quantize", variant="primary", interactive=gguf_utils is not None)
            if 'LogsView' in globals() and LogsView != gr.Textbox:
                gguf_logs_output = LogsView(label="GGUF Conversion Logs", lines=15, elem_classes=log_elem_class)
            else:
                gguf_logs_output = gr.Textbox(label="GGUF Conversion Logs", lines=15, interactive=False, elem_classes=log_elem_class)
            gguf_final_status_display = gr.Markdown()
            gguf_output_image = gr.Image(show_label=False)

    # --- Event Handler Function Definitions ---
    def update_gguf_src_visibility(choice, merged_path_val):
        is_hf = (choice == "HF Hub")
        is_local = (choice == "Local Path")
        is_merged = (choice == "Output from Merge Step")

        merged_display_value = "N/A"
        if is_merged:
            if merged_path_val and Path(str(merged_path_val)).exists(): # Ensure merged_path_val is stringified for Path
                merged_display_value = f"Using: {merged_path_val}"
            else:
                merged_display_value = "Warning: Merged path from Step 1 is invalid, not found, or Step 1 not yet run successfully."
        
        return {
            gguf_hf_group: gr.update(visible=is_hf),
            gguf_local_model_path: gr.update(visible=is_local),
            gguf_merged_model_display: gr.update(visible=is_merged, value=merged_display_value)
        }

    # --- Main Handler Function Definitions ---
    def handle_merge_models(yaml_config, hf_token_merge_input, repo_name_merge, local_save_path, use_gpu_for_merge, use_for_gguf, current_gguf_model_source_choice):
        if not mergekit_utils:
            error_msg = "Mergekit utils not loaded."
            log_entry = [Log(error_msg, "ERROR", datetime.datetime.now())] if LogsView != gr.Textbox else error_msg
            # Update GGUF source visibility as merged path will be None
            gguf_updates = update_gguf_src_visibility(current_gguf_model_source_choice, None)
            final_yield_data = {merge_status_output: error_msg, merge_logs_output: log_entry, merged_model_path_state: None}
            final_yield_data.update(gguf_updates)
            yield final_yield_data
            return

        effective_hf_token_for_merge = hf_token_merge_input or HF_TOKEN
        output_path_for_mergekit_process = local_save_path # Can be None or empty
        persistent_passthrough_dir_for_gguf = None # Path object

        # If using for GGUF and no specific local save path is given, create a temp persistent dir
        if use_for_gguf and not local_save_path:
            # TEMP_DIR_ROOT is used here, and will reflect the APP_TEMP_ROOT env var if set
            dir_name = f"merged_passthrough_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            persistent_passthrough_dir_for_gguf = TEMP_DIR_ROOT / dir_name # [cite: 1]
            persistent_passthrough_dir_for_gguf.mkdir(parents=True, exist_ok=True)
            output_path_for_mergekit_process = str(persistent_passthrough_dir_for_gguf)
        elif local_save_path: # If a local save path is specified, ensure it exists
            Path(local_save_path).mkdir(parents=True, exist_ok=True)
            # output_path_for_mergekit_process is already set to local_save_path

        # Check if any output mechanism is specified
        if not output_path_for_mergekit_process and not repo_name_merge:
            error_msg = "No local save path, HF repository name for upload, or GGUF passthrough option selected. Please specify an output for the merge."
            log_entry = [Log(error_msg, "ERROR", datetime.datetime.now())] if LogsView != gr.Textbox else error_msg
            gguf_updates = update_gguf_src_visibility(current_gguf_model_source_choice, None)
            final_yield_data = {merge_status_output: f"### ❌ Config Error\n{error_msg}", merge_logs_output: log_entry, merged_model_path_state: None}
            final_yield_data.update(gguf_updates)
            yield final_yield_data
            return
        
        # Check for HF token if repo name is specified (and not community upload without specific token)
        community_token = getattr(mergekit_utils, 'community_hf_token_val', None) if mergekit_utils else None
        if repo_name_merge and not effective_hf_token_for_merge and not (repo_name_merge.startswith("mergekit-community/") and community_token):
            error_msg = "HF Repo specified for merge, but no effective HF token (neither in field nor HF_TOKEN env var)."
            log_entry = [Log(error_msg, "ERROR", datetime.datetime.now())] if LogsView != gr.Textbox else error_msg
            gguf_updates = update_gguf_src_visibility(current_gguf_model_source_choice, None)
            final_yield_data = {merge_status_output: f"### ❌ Merge Error\n{error_msg}", merge_logs_output: log_entry, merged_model_path_state: None}
            final_yield_data.update(gguf_updates)
            yield final_yield_data
            return

        log_q, result_container = queue.Queue(), {} # type: ignore
        accumulated_logs = [] if LogsView != gr.Textbox else ""

        def log_callback_for_thread(raw_log_message, level_arg="INFO"):
            message_content = str(raw_log_message) # Ensure it's a string
            # Basic prefix parsing for level, can be expanded
            parsed_level = "INFO" # Default level
            found_prefix = False
            prefix_map = [
                ("[STDOUT]", "INFO"), ("[STDERR]", "ERROR"),
                ("[DEBUG]", "DEBUG"), ("[INFO]", "INFO"),
                ("[WARNING]", "WARNING"), ("[ERROR]", "ERROR")
            ]
            for p_str, p_lvl in prefix_map:
                if message_content.startswith(p_str):
                    parsed_level = p_lvl
                    message_content = message_content[len(p_str):].lstrip()
                    found_prefix = True
                    break
            
            if not found_prefix and level_arg: # Fallback to explicit level_arg if no prefix found
                parsed_level = level_arg.upper()

            if message_content or parsed_level == "ERROR": # Log empty messages only if they are errors
                log_q.put(Log(message_content, parsed_level, datetime.datetime.now()))

        merge_thread_instance = None # type: ignore
        def merge_thread_target():
            try:
                # TEMP_DIR_ROOT is passed as temp_dir_base. mergekit_utils will check MERGEKIT_JOB_TEMP_DIR env var first.
                _, final_path_from_util, error_msg_from_util = mergekit_utils.process_model_merge( # [cite: 4]
                    yaml_config_str=yaml_config, hf_token_merge=effective_hf_token_for_merge,
                    repo_name=repo_name_merge, local_path_merge_output=output_path_for_mergekit_process, # This is where mergekit saves its output
                    community_hf_token_val=community_token, use_gpu_bool=use_gpu_for_merge,
                    temp_dir_base=str(TEMP_DIR_ROOT), log_fn=log_callback_for_thread # [cite: 1]
                )
                result_container['final_path'] = final_path_from_util
                result_container['error_msg'] = error_msg_from_util
            except Exception as e_thread:
                err_msg = f"Critical error in merge thread: {str(e_thread)}"
                result_container['error_msg'] = err_msg
                log_callback_for_thread(err_msg, "CRITICAL_ERROR") # Use a distinct level for critical thread failures
            finally:
                log_q.put(None) # Sentinel to indicate thread completion

        merge_thread_instance = Thread(target=merge_thread_target, name="MergeThread")
        merge_thread_instance.start()
        initial_log_message = "Initiating merge process..."
        if 'LogsView' in globals() and LogsView != gr.Textbox:
            initial_log = Log(initial_log_message, "INFO", datetime.datetime.now())
            accumulated_logs.append(initial_log)
            initial_log_display = accumulated_logs.copy()
        else:
            accumulated_logs = initial_log_message + "\n"
            initial_log_display = accumulated_logs
        
        # Initial yield to show logs started
        yield {merge_status_output: "Starting merge...", merge_logs_output: initial_log_display, merged_model_path_state: None}

        while True:
            try:
                log_item = log_q.get(timeout=0.1) # Timeout to allow checking thread status
                if log_item is None: # Sentinel received
                    break
                if 'LogsView' in globals() and LogsView != gr.Textbox:
                    accumulated_logs.append(log_item)
                    new_logs_update = accumulated_logs.copy() # Use a copy for safety if LogsView modifies it
                else: # Textbox fallback
                    log_line = f"[{log_item.timestamp.strftime('%H:%M:%S')}] [{log_item.level}] {log_item.message}\n"
                    accumulated_logs += log_line
                    new_logs_update = accumulated_logs
                yield {merge_logs_output: new_logs_update}
            except queue.Empty:
                if not merge_thread_instance.is_alive(): # Thread finished unexpectedly or normally before sentinel
                    break
                pass # Continue waiting for logs or thread completion
        
        merge_thread_instance.join(timeout=10) # Wait for thread to ensure it's fully finished

        final_path, err_msg = result_container.get('final_path'), result_container.get('error_msg')
        status_md, gguf_path_state = "", None

        if not err_msg and final_path:
            status_md = f"### ✅ Merge Successful!\nOutput: `{final_path}`"
            if use_for_gguf:
                gguf_path_state = final_path # This is the path to the merged model
                status_md += "\nPath set for GGUF conversion."
        elif err_msg:
            status_md = f"### ❌ Merge Failed\n{err_msg}"
            # If merge failed and we used a temporary passthrough directory, clean it up
            if persistent_passthrough_dir_for_gguf and persistent_passthrough_dir_for_gguf.exists():
                try:
                    shutil.rmtree(persistent_passthrough_dir_for_gguf)
                    log_callback_for_thread(f"Cleaned up temporary passthrough directory: {persistent_passthrough_dir_for_gguf}", "INFO")
                except Exception as e_clean:
                    log_callback_for_thread(f"Error cleaning passthrough directory {persistent_passthrough_dir_for_gguf}: {e_clean}", "WARNING")
        else: # Should not happen if thread ran, but as a fallback
            status_md = "Merge outcome unclear. Check logs."

        # Final log entry
        end_log_msg = (f"Merge finished with errors: {err_msg}" if err_msg else "Merge process finished.")
        end_log_lvl = ("ERROR" if err_msg else "INFO")
        if 'LogsView' in globals() and LogsView != gr.Textbox:
            final_log = Log(end_log_msg, end_log_lvl, datetime.datetime.now())
            accumulated_logs.append(final_log)
            final_log_disp = accumulated_logs.copy()
        else:
            accumulated_logs += f"[{datetime.datetime.now().strftime('%H:%M:%S')}] [{end_log_lvl}] {end_log_msg}\n"
            final_log_disp = accumulated_logs
        
        final_yield_updates = {
            merge_status_output: status_md,
            merge_logs_output: final_log_disp,
            merged_model_path_state: gguf_path_state # This state is used by GGUF step
        }
        # Update GGUF tab based on the outcome
        gguf_tab_updates = update_gguf_src_visibility(current_gguf_model_source_choice, gguf_path_state)
        final_yield_updates.update(gguf_tab_updates)
        yield final_yield_updates


    def handle_gguf_conversion(
        model_source_choice, current_merged_model_path, hf_model_id_input, local_model_path_input,
        gguf_custom_name,
        gguf_hf_token_val,
        quantization_methods, use_importance_matrix, imatrix_quant_methods,
        use_bundled_imatrix_checkbox_value, 
        upload_to_huggingface, make_repo_private, local_gguf_output_path,
        should_split_model, split_max_tensors_val, split_max_size_val
    ):
        if not gguf_utils:
            error_msg = "GGUF utils not loaded."
            log_entry = [Log(error_msg, "ERROR", datetime.datetime.now())] if LogsView != gr.Textbox else error_msg # type: ignore
            yield {gguf_logs_output: log_entry, gguf_final_status_display: f"### ❌ Error\n{error_msg}", gguf_output_image: None}
            return

        effective_hf_token_for_gguf = gguf_hf_token_val or HF_TOKEN

        log_q_gguf, result_container_gguf = queue.Queue(), {} # type: ignore
        actual_model_source_for_util, model_id_for_util, local_model_path_for_util, error_message_setup = None, None, None, None

        # Determine model source for gguf_utils
        if model_source_choice == "Output from Merge Step":
            if not current_merged_model_path or not Path(str(current_merged_model_path)).exists(): # Ensure path is valid
                error_message_setup = "Merged model path from Step 1 is invalid or does not exist. Please run Step 1 successfully."
            else:
                actual_model_source_for_util, local_model_path_for_util = "Local Path", str(current_merged_model_path)
        elif model_source_choice == "HF Hub":
            if not hf_model_id_input: error_message_setup = "Hugging Face Model ID is required."
            else: actual_model_source_for_util, model_id_for_util = "HF Hub", hf_model_id_input
        elif model_source_choice == "Local Path":
            if not local_model_path_input or not Path(local_model_path_input).exists(): # Ensure path is valid
                error_message_setup = "Specified local model path is invalid or does not exist."
            else: actual_model_source_for_util, local_model_path_for_util = "Local Path", str(local_model_path_input)
        else:
            error_message_setup = "Invalid GGUF model source selected."

        if error_message_setup:
            log_entry = [Log(error_message_setup, "ERROR", datetime.datetime.now())] if LogsView != gr.Textbox else error_message_setup # type: ignore
            yield {gguf_logs_output: log_entry, gguf_final_status_display: f"### ❌ Config Error\n{error_message_setup}", gguf_output_image: None}
            return

        # TEMP_DIR_ROOT will reflect APP_TEMP_ROOT env var if set
        effective_local_out_path = local_gguf_output_path or str(TEMP_DIR_ROOT / "gguf_exports_default") # [cite: 1]
        Path(effective_local_out_path).mkdir(parents=True, exist_ok=True)
        
        train_data_path_for_util = None
        # Logger for this scope, defined before its first potential use by the imatrix logic
        _accumulated_logs_gguf_local_init = [] if LogsView != gr.Textbox else "" # type: ignore
        
        def log_callback_for_gguf_thread_init_phase(raw_log_message, level_arg="INFO"):
            # This function's main purpose here is to be available for the imatrix path decision logic
            message_content, parsed_level = str(raw_log_message), "INFO"
            known_direct_levels = ["INFO", "DEBUG", "WARNING", "ERROR", "CRITICAL_ERROR"]
            if level_arg and level_arg.upper() in known_direct_levels:
                parsed_level = level_arg.upper()

        if use_importance_matrix:
            if use_bundled_imatrix_checkbox_value:
                if Path(BUNDLED_IMATRIX_PATH).is_file(): # [cite: 1]
                    train_data_path_for_util = BUNDLED_IMATRIX_PATH # [cite: 1]
                    log_callback_for_gguf_thread_init_phase(f"Using bundled imatrix file: {train_data_path_for_util}", "INFO")
                else:
                    log_callback_for_gguf_thread_init_phase(f"WARNING: Bundled imatrix file '{BUNDLED_IMATRIX_PATH}' selected but not found. Proceeding without it.", "WARNING") # [cite: 1]
                    train_data_path_for_util = None # Explicitly set to None
            else: # Importance matrix enabled, but bundled file *not* selected
                log_callback_for_gguf_thread_init_phase("INFO: Importance matrix enabled, but the option to use the bundled 'groups_merged.txt' was not selected. No specific imatrix training data file will be passed to gguf_utils unless it handles a default internally.", "INFO")
                # train_data_path_for_util remains None, which is the correct state here.
        
        accumulated_logs_gguf = [] if LogsView != gr.Textbox else "" # type: ignore

        def log_callback_for_gguf_thread(raw_log_message, level_arg="INFO"):
            message_content, parsed_level = str(raw_log_message), "INFO"
            known_direct_levels = ["INFO", "DEBUG", "WARNING", "ERROR", "CRITICAL_ERROR"]
            if level_arg and level_arg.upper() in known_direct_levels:
                parsed_level = level_arg.upper()
            
            if message_content or parsed_level in ["ERROR", "CRITICAL_ERROR"]: # Log empty messages only if critical
                log_q_gguf.put(Log(message_content, parsed_level, datetime.datetime.now()))
        
        # Re-log imatrix decision using the proper threaded logger if conditions met
        # This ensures the log appears in the Gradio UI log view.
        if use_importance_matrix:
            if use_bundled_imatrix_checkbox_value:
                if Path(BUNDLED_IMATRIX_PATH).is_file(): # [cite: 1]
                    # train_data_path_for_util is already set
                    log_callback_for_gguf_thread(f"Confirmed: Using bundled imatrix file for GGUF conversion: {train_data_path_for_util}", "INFO")
                else:
                    log_callback_for_gguf_thread(f"WARNING: Bundled imatrix file '{BUNDLED_IMATRIX_PATH}' was selected but not found at execution time. Proceeding without it.", "WARNING") # [cite: 1]
            elif train_data_path_for_util is None: # Only log if it wasn't set (e.g. by a hypothetical previous custom file input)
                 log_callback_for_gguf_thread("INFO: Importance matrix enabled, bundled imatrix not selected. No specific imatrix training data file passed for GGUF conversion.", "INFO")


        gguf_thread_instance = None # type: ignore
        def gguf_conversion_thread_target():
            try:
                if not hasattr(gguf_utils, 'process_gguf_conversion'): # [cite: 1]
                    raise GGUFConversionError("process_gguf_conversion function is missing from gguf_utils.")

                # TEMP_DIR_ROOT will reflect APP_TEMP_ROOT env var if set
                gguf_utils.process_gguf_conversion( # [cite: 1, 3]
                    actual_model_source_for_util,
                    effective_hf_token_for_gguf,
                    log_callback_for_gguf_thread, result_container_gguf,
                    model_id_for_util, local_model_path_for_util,
                    gguf_custom_name, # Passed to gguf_utils [cite: 3]
                    quantization_methods, use_importance_matrix,
                    imatrix_quant_methods, upload_to_huggingface, make_repo_private,
                    train_data_path_for_util, # This is now correctly set based on checkbox and file existence [cite: 3]
                    str(effective_local_out_path), should_split_model,
                    int(split_max_tensors_val) if split_max_tensors_val is not None else 256, # Ensure int conversion
                    split_max_size_val,
                    str(TEMP_DIR_ROOT / "gguf_temps"), str(TEMP_DIR_ROOT / "gguf_downloads") # [cite: 1]
                )
            except Exception as e:
                # Ensure any exception from gguf_utils or this thread is caught and logged
                tb_str = traceback.format_exc()
                result_container_gguf['error_msg'] = f"GGUF conversion thread error: {str(e)}"
                log_callback_for_gguf_thread(f"Critical GGUF Thread Error: {str(e)}\nTraceback:\n{tb_str}", "CRITICAL_ERROR")
            finally:
                log_q_gguf.put(None) # Sentinel for completion

        gguf_thread_instance = Thread(target=gguf_conversion_thread_target, name="GGUFConversionThread")
        gguf_thread_instance.start()
        initial_log_message_gguf = "Initiating GGUF conversion and quantization..."
        if 'LogsView' in globals() and LogsView != gr.Textbox:
            initial_log_gguf = Log(initial_log_message_gguf, "INFO", datetime.datetime.now())
            accumulated_logs_gguf.append(initial_log_gguf)
            initial_log_display_gguf = accumulated_logs_gguf.copy()
        else:
            accumulated_logs_gguf = initial_log_message_gguf + "\n"
            initial_log_display_gguf = accumulated_logs_gguf
        
        yield {gguf_logs_output: initial_log_display_gguf, gguf_final_status_display: "Starting GGUF conversion...", gguf_output_image: None}

        while True:
            try:
                log_item_gguf = log_q_gguf.get(timeout=0.1)
                if log_item_gguf is None: # Sentinel
                    break
                if 'LogsView' in globals() and LogsView != gr.Textbox:
                    accumulated_logs_gguf.append(log_item_gguf)
                    new_logs_update_gguf = accumulated_logs_gguf.copy()
                else: # Textbox fallback
                    log_line_gguf = f"[{log_item_gguf.timestamp.strftime('%H:%M:%S')}] [{log_item_gguf.level}] {log_item_gguf.message}\n"
                    accumulated_logs_gguf += log_line_gguf
                    new_logs_update_gguf = accumulated_logs_gguf
                yield {gguf_logs_output: new_logs_update_gguf}
            except queue.Empty:
                if not gguf_thread_instance.is_alive():
                    break # Thread finished
                pass # Continue polling
        
        gguf_thread_instance.join(timeout=10) # Ensure thread is fully cleaned up

        html_out, img_path, err_msg = result_container_gguf.get('final_html',""), result_container_gguf.get('image_path'), result_container_gguf.get('error_msg')
        status_md, img_update = "", gr.update(value=None) # Initialize img_update

        end_log_msg = "GGUF conversion process completed."
        end_log_lvl = "INFO"

        if err_msg:
            status_md = f"### ❌ GGUF Conversion Failed\n<pre>{str(err_msg)}</pre>" # Use <pre> for better formatting of error messages
            end_log_msg = f"GGUF conversion finished with errors: {str(err_msg)}"
            end_log_lvl = "ERROR"
        elif html_out:
            status_md = html_out # This should come from gguf_utils
        else: # Fallback if no specific HTML or error
            status_md = "GGUF conversion finished. Review logs for details."

        if img_path and Path(img_path).exists():
            img_update = gr.update(value=str(img_path)) # Ensure path is string
        elif img_path: # Image path provided but file doesn't exist
            img_warn_msg = f"Output image path '{img_path}' was provided by gguf_utils, but the image file was not found."
            log_callback_for_gguf_thread(img_warn_msg, "WARNING") # Log this warning
            # No need to append to accumulated_logs_gguf here if using the thread logger

        # Final log to Gradio UI
        if 'LogsView' in globals() and LogsView != gr.Textbox:
            final_log_gguf = Log(end_log_msg, end_log_lvl, datetime.datetime.now())
            accumulated_logs_gguf.append(final_log_gguf)
            final_log_disp_gguf = accumulated_logs_gguf.copy()
        else:
            accumulated_logs_gguf += f"[{datetime.datetime.now().strftime('%H:%M:%S')}] [{end_log_lvl}] {end_log_msg}\n"
            final_log_disp_gguf = accumulated_logs_gguf
            
        yield {gguf_logs_output: final_log_disp_gguf, gguf_final_status_display: status_md, gguf_output_image: img_update}
    
    # --- Event Handlers LINKING (after UI components are defined) ---
    demo.load(populate_examples_for_dropdown, outputs=[example_dropdown])
    load_example_btn.click(load_selected_example_content, inputs=[example_dropdown], outputs=[merge_yaml_config])

    merge_button.click(handle_merge_models,
        inputs=[
            merge_yaml_config, merge_hf_token_input, merge_repo_name,
            merge_local_save_path, merge_use_gpu, merge_use_for_gguf,
            gguf_model_source # Pass current GGUF source choice to update its display if merge fails/succeeds
        ],
        outputs=[
            merge_status_output, merge_logs_output, merged_model_path_state,
            # Outputs to update GGUF tab's source visibility
            gguf_hf_group, gguf_local_model_path, gguf_merged_model_display
        ])

    gguf_convert_btn.click(handle_gguf_conversion,
        inputs=[
            gguf_model_source, merged_model_path_state, gguf_model_id, gguf_local_model_path,
            gguf_custom_name_input,
            gguf_hf_token_input,
            gguf_q_methods, gguf_use_imatrix, gguf_imatrix_q_methods,
            gguf_use_bundled_imatrix_checkbox, # Pass the new checkbox component's value [cite: 1]
            gguf_upload_to_hf, gguf_private_repo, gguf_local_output_path,
            gguf_split_model, gguf_split_max_tensors, gguf_split_max_size
        ],
        outputs=[gguf_logs_output, gguf_final_status_display, gguf_output_image])

    # Update GGUF source display when merged_model_path_state changes (e.g., after merge)
    merged_model_path_state.change(update_gguf_src_visibility,
                                    inputs=[gguf_model_source, merged_model_path_state],
                                    outputs=[gguf_hf_group, gguf_local_model_path, gguf_merged_model_display])

    # Update GGUF source display when the radio button choice changes
    gguf_model_source.change(update_gguf_src_visibility,
                                inputs=[gguf_model_source, merged_model_path_state],
                                outputs=[gguf_hf_group, gguf_local_model_path, gguf_merged_model_display])

    # Initial UI state updates on demo load
    def initial_load_updates(merged_path_on_load, src_choice_on_load, imatrix_val_on_load, split_val_on_load, upload_val_on_load):
        updates = update_gguf_src_visibility(src_choice_on_load, merged_path_on_load) # Initial GGUF source visibility
        updates.update({
            gguf_q_methods: gr.update(visible=not imatrix_val_on_load),
            gguf_imatrix_q_methods: gr.update(visible=imatrix_val_on_load),
            gguf_use_bundled_imatrix_checkbox: gr.update(visible=imatrix_val_on_load), # visibility for new checkbox [cite: 1]
            gguf_split_max_tensors: gr.update(visible=split_val_on_load),
            gguf_split_max_size: gr.update(visible=split_val_on_load),
            gguf_private_repo: gr.update(visible=upload_val_on_load)
        })
        return updates

    demo.load(initial_load_updates,
        inputs=[merged_model_path_state, gguf_model_source, gguf_use_imatrix, gguf_split_model, gguf_upload_to_hf],
        outputs=[
            gguf_hf_group, gguf_local_model_path, gguf_merged_model_display, # For GGUF source visibility
            gguf_q_methods, gguf_imatrix_q_methods, 
            gguf_use_bundled_imatrix_checkbox, # For new checkbox visibility [cite: 1]
            gguf_split_max_tensors, gguf_split_max_size, gguf_private_repo # Other conditional visibilities
        ])
    
    # Visibility toggles for quantization methods and bundled imatrix checkbox based on gguf_use_imatrix
    gguf_use_imatrix.change(
        lambda use_imatrix_checked: (
            gr.update(visible=not use_imatrix_checked), # gguf_q_methods
            gr.update(visible=use_imatrix_checked),   # gguf_imatrix_q_methods
            gr.update(visible=use_imatrix_checked)    # gguf_use_bundled_imatrix_checkbox [cite: 1]
        ),
        inputs=gguf_use_imatrix,
        outputs=[gguf_q_methods, gguf_imatrix_q_methods, gguf_use_bundled_imatrix_checkbox], # Target new checkbox [cite: 1]
        api_name=False
    )
    # Visibility toggles for GGUF splitting options
    gguf_split_model.change(
        lambda split_checked: (gr.update(visible=split_checked), gr.update(visible=split_checked)),
        inputs=gguf_split_model,
        outputs=[gguf_split_max_tensors, gguf_split_max_size],
        api_name=False
    )
    # Visibility toggle for private repo option based on upload to HF
    gguf_upload_to_hf.change(
        lambda upload_checked: gr.update(visible=upload_checked),
        inputs=gguf_upload_to_hf,
        outputs=[gguf_private_repo],
        api_name=False
    )

if __name__ == "__main__":
    # Clean up older temporary folders from mergekit if any exist from previous runs
    if mergekit_utils and hasattr(mergekit_utils, 'clean_tmp_folders'):
        try:
            cleaned_count = mergekit_utils.clean_tmp_folders() # [cite: 4]
            if cleaned_count > 0:
                print(f"Cleaned up {cleaned_count} old mergekit temporary item(s).")
        except Exception as e_clean:
            print(f"Error during initial cleanup of mergekit temp folders: {e_clean}")
            
    demo.queue(default_concurrency_limit=2, max_size=20).launch(debug=True, show_api=False)