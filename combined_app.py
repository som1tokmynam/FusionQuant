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
TEMP_DIR_ROOT = Path("outputs/combined_app_temp")
TEMP_DIR_ROOT.mkdir(parents=True, exist_ok=True)

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
        example_filenames = mergekit_utils.get_example_yaml_filenames_for_gr_examples()
        labeled_examples = []
        if not example_filenames or (len(example_filenames) == 1 and "default_example.yaml" in example_filenames[0]):
            default_content = mergekit_utils.get_example_yaml_content("default_example.yaml")
            labeled_examples.append([extract_example_label(None, 0, "Default Example.yaml"), default_content])
        else:
            for i, filename in enumerate(example_filenames):
                if filename == "default_example.yaml" and len(example_filenames) > 1:
                    if not any(f != "default_example.yaml" for f in example_filenames):
                        pass
                    else: continue
                content = mergekit_utils.get_example_yaml_content(filename)
                if content.startswith("# Error reading"): continue
                labeled_examples.append([extract_example_label(None, i, filename), content])
        if not labeled_examples: # Fallback if all examples failed to load but default exists
            default_content = mergekit_utils.get_example_yaml_content("default_example.yaml")
            labeled_examples.append([extract_example_label(None, 0, "Default Example.yaml"), default_content])
        return labeled_examples
    except Exception as e:
        print(f"Error loading merge examples: {e}")
        default_content = "models:\n  - model: cognitivecomputations/dolphin-2.6-mistral-7b-dpo-laser\n    parameters:\n      weight: 0.5\n  - model: OpenPipe/mistral-ft-optimized-1218\n    parameters:\n      weight: 0.5\nmerge_method: linear\ndtype: float16"
        return [[extract_example_label(None, 0, "Fallback Linear Merge.yaml"), default_content]]

# --- Gradio Callbacks for Initial UI Population (Defined before .load calls) ---
def populate_examples_for_dropdown():
    examples = load_merge_examples()
    choices = [example[0] for example in examples]
    # Use gr.update() for component updates
    return gr.update(choices=choices, value=choices[0] if choices else None)

def load_selected_example_content(selected_label):
    if not selected_label: return ""
    examples = load_merge_examples()
    for label, content in examples:
        if label == selected_label: return content
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
                    if mergekit_utils and hasattr(mergekit_utils, 'create_example_files'):
                        try: mergekit_utils.create_example_files()
                        except Exception as e: print(f"Error creating merge examples: {e}")
                    gr.Markdown("### Load Merge Example")
                    with gr.Row():
                        example_dropdown = gr.Dropdown(choices=[], label="Select Example", interactive=True)
                        load_example_btn = gr.Button("Load Example", size="sm")
                with gr.Column(scale=1):
                    merge_hf_token_input = gr.Textbox(label="HF Write Token (Mergekit upload)", type="password", placeholder="Uses HF_TOKEN env var if blank.")
                    merge_repo_name = gr.Textbox(label="HF Repo Name (Mergekit upload)", placeholder="e.g., YourUser/MyMerge")
                    merge_local_save_path = gr.Textbox(label="Local Save Path (Merged Model)", placeholder=f"e.g., {TEMP_DIR_ROOT}/my-merged-model")
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
            gguf_train_data_file = gr.File(label="Training Data (Imatrix)", file_types=[".txt"], visible=False)

            gr.Markdown("### Output Settings")
            gguf_custom_name_input = gr.Textbox(label="Custom GGUF Base Name (Optional)", placeholder="e.g., MyAwesomeModel-7B")
            gguf_upload_to_hf = gr.Checkbox(label="Upload GGUF to HF", value=True)
            gguf_private_repo = gr.Checkbox(label="Make GGUF Repo Private", value=False, visible=True)
            gguf_local_output_path = gr.Textbox(label="Local Save Path (GGUFs)", placeholder=f"e.g., {TEMP_DIR_ROOT}/gguf_exports")
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
            if merged_path_val and Path(str(merged_path_val)).exists():
                merged_display_value = f"Using: {merged_path_val}"
            else:
                merged_display_value = "Warning: Merged path from Step 1 is invalid, not found, or Step 1 not yet run successfully."
        # Use gr.update() for component updates
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
            gguf_updates = update_gguf_src_visibility(current_gguf_model_source_choice, None)
            final_yield_data = {merge_status_output: error_msg, merge_logs_output: log_entry, merged_model_path_state: None}
            final_yield_data.update(gguf_updates)
            yield final_yield_data
            return

        effective_hf_token_for_merge = hf_token_merge_input or HF_TOKEN
        output_path_for_mergekit_process = local_save_path
        persistent_passthrough_dir_for_gguf = None

        if use_for_gguf and not local_save_path:
            dir_name = f"merged_passthrough_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            persistent_passthrough_dir_for_gguf = TEMP_DIR_ROOT / dir_name
            persistent_passthrough_dir_for_gguf.mkdir(parents=True, exist_ok=True)
            output_path_for_mergekit_process = str(persistent_passthrough_dir_for_gguf)
        elif local_save_path:
            Path(local_save_path).mkdir(parents=True, exist_ok=True)

        if not output_path_for_mergekit_process and not repo_name_merge:
            error_msg = "No local save, HF repo, or GGUF passthrough. Specify output."
            log_entry = [Log(error_msg, "ERROR", datetime.datetime.now())] if LogsView != gr.Textbox else error_msg
            gguf_updates = update_gguf_src_visibility(current_gguf_model_source_choice, None)
            final_yield_data = {merge_status_output: f"### ❌ Config Error\n{error_msg}", merge_logs_output: log_entry, merged_model_path_state: None}
            final_yield_data.update(gguf_updates)
            yield final_yield_data
            return

        community_token = getattr(mergekit_utils, 'community_hf_token_val', None) if mergekit_utils else None
        if repo_name_merge and not effective_hf_token_for_merge and not (repo_name_merge.startswith("mergekit-community/") and community_token):
            error_msg = "HF Repo specified for merge, but no effective HF token (either in field or HF_TOKEN env var)."
            log_entry = [Log(error_msg, "ERROR", datetime.datetime.now())] if LogsView != gr.Textbox else error_msg
            gguf_updates = update_gguf_src_visibility(current_gguf_model_source_choice, None)
            final_yield_data = {merge_status_output: f"### ❌ Merge Error\n{error_msg}", merge_logs_output: log_entry, merged_model_path_state: None}
            final_yield_data.update(gguf_updates)
            yield final_yield_data
            return

        log_q, result_container = queue.Queue(), {}
        accumulated_logs = [] if LogsView != gr.Textbox else ""

        def log_callback_for_thread(raw_log_message, level_arg="INFO"):
            message_content = str(raw_log_message)
            parsed_level = "INFO"; found_prefix = False
            prefix_map = [("[STDOUT]", "INFO"), ("[STDERR]", "ERROR"), ("[DEBUG]", "DEBUG"), ("[INFO]", "INFO"), ("[WARNING]", "WARNING"), ("[ERROR]", "ERROR")]
            for p_str, p_lvl in prefix_map:
                if message_content.startswith(p_str): parsed_level, message_content, found_prefix = p_lvl, message_content[len(p_str):].lstrip(), True; break
            if not found_prefix and level_arg: parsed_level = level_arg.upper()
            if message_content or parsed_level == "ERROR": log_q.put(Log(message_content, parsed_level, datetime.datetime.now()))

        merge_thread_instance = None
        def merge_thread_target():
            try:
                _, final_path_from_util, error_msg_from_util = mergekit_utils.process_model_merge(
                    yaml_config_str=yaml_config, hf_token_merge=effective_hf_token_for_merge,
                    repo_name=repo_name_merge, local_path_merge_output=output_path_for_mergekit_process,
                    community_hf_token_val=community_token, use_gpu_bool=use_gpu_for_merge,
                    temp_dir_base=str(TEMP_DIR_ROOT), log_fn=log_callback_for_thread
                )
                result_container['final_path'] = final_path_from_util
                result_container['error_msg'] = error_msg_from_util
            except Exception as e_thread:
                err_msg = f"Critical error in merge thread: {str(e_thread)}"
                result_container['error_msg'] = err_msg
                log_callback_for_thread(err_msg, "CRITICAL_ERROR")
            finally: log_q.put(None)

        merge_thread_instance = Thread(target=merge_thread_target, name="MergeThread")
        merge_thread_instance.start()
        initial_log_message = "Initiating merge..."
        if 'LogsView' in globals() and LogsView != gr.Textbox:
            initial_log = Log(initial_log_message, "INFO", datetime.datetime.now()); accumulated_logs.append(initial_log)
            initial_log_display = accumulated_logs.copy()
        else:
            accumulated_logs = initial_log_message + "\n"; initial_log_display = accumulated_logs

        yield {merge_status_output: "Starting merge...", merge_logs_output: initial_log_display, merged_model_path_state: None}

        while True:
            try:
                log_item = log_q.get(timeout=0.1)
                if log_item is None: break
                if 'LogsView' in globals() and LogsView != gr.Textbox:
                    accumulated_logs.append(log_item); new_logs_update = accumulated_logs.copy()
                else:
                    log_line = f"[{log_item.timestamp.strftime('%H:%M:%S')}] [{log_item.level}] {log_item.message}\n"
                    accumulated_logs += log_line; new_logs_update = accumulated_logs
                yield {merge_logs_output: new_logs_update}
            except queue.Empty:
                if not merge_thread_instance.is_alive(): break
                pass
        merge_thread_instance.join(timeout=10)

        final_path, err_msg = result_container.get('final_path'), result_container.get('error_msg')
        status_md, gguf_path_state = "", None
        if not err_msg and final_path:
            status_md = f"### ✅ Merge Successful!\nOutput: `{final_path}`"
            if use_for_gguf: gguf_path_state = final_path; status_md += "\nPath set for GGUF."
        elif err_msg:
            status_md = f"### ❌ Merge Failed\n{err_msg}"
            if persistent_passthrough_dir_for_gguf:
                try: shutil.rmtree(persistent_passthrough_dir_for_gguf)
                except Exception as e_clean: log_callback_for_thread(f"Error cleaning passthrough dir: {e_clean}", "WARNING")
        else: status_md = "Merge outcome unclear."

        end_log_msg, end_log_lvl = (f"Merge finished with errors: {err_msg}" if err_msg else "Merge finished."), ("ERROR" if err_msg else "INFO")
        if 'LogsView' in globals() and LogsView != gr.Textbox:
            final_log = Log(end_log_msg, end_log_lvl, datetime.datetime.now()); accumulated_logs.append(final_log)
            final_log_disp = accumulated_logs.copy()
        else:
            accumulated_logs += f"[{datetime.datetime.now().strftime('%H:%M:%S')}] [{end_log_lvl}] {end_log_msg}\n"; final_log_disp = accumulated_logs

        final_yield_updates = {
            merge_status_output: status_md,
            merge_logs_output: final_log_disp,
            merged_model_path_state: gguf_path_state
        }
        gguf_tab_updates = update_gguf_src_visibility(current_gguf_model_source_choice, gguf_path_state)
        final_yield_updates.update(gguf_tab_updates)
        yield final_yield_updates


    def handle_gguf_conversion(
        model_source_choice, current_merged_model_path, hf_model_id_input, local_model_path_input,
        gguf_custom_name,
        gguf_hf_token_val,
        quantization_methods, use_importance_matrix, imatrix_quant_methods, training_data_file,
        upload_to_huggingface, make_repo_private, local_gguf_output_path,
        should_split_model, split_max_tensors_val, split_max_size_val
    ):
        if not gguf_utils:
            error_msg = "GGUF utils not loaded."
            log_entry = [Log(error_msg, "ERROR", datetime.datetime.now())] if LogsView != gr.Textbox else error_msg
            yield {gguf_logs_output: log_entry, gguf_final_status_display: f"### ❌ Error\n{error_msg}", gguf_output_image: None}
            return

        effective_hf_token_for_gguf = gguf_hf_token_val or HF_TOKEN

        log_q_gguf, result_container_gguf = queue.Queue(), {}
        actual_model_source_for_util, model_id_for_util, local_model_path_for_util, error_message_setup = None, None, None, None

        if model_source_choice == "Output from Merge Step":
            if not current_merged_model_path or not Path(str(current_merged_model_path)).exists():
                error_message_setup = "Merged model path invalid or DNE. Run Step 1."
            else: actual_model_source_for_util, local_model_path_for_util = "Local Path", str(current_merged_model_path)
        elif model_source_choice == "HF Hub":
            if not hf_model_id_input: error_message_setup = "HF Model ID required."
            else: actual_model_source_for_util, model_id_for_util = "HF Hub", hf_model_id_input
        elif model_source_choice == "Local Path":
            if not local_model_path_input or not Path(local_model_path_input).exists(): error_message_setup = "Local model path invalid or DNE."
            else: actual_model_source_for_util, local_model_path_for_util = "Local Path", str(local_model_path_input)
        else: error_message_setup = "Invalid GGUF source."

        if error_message_setup:
            log_entry = [Log(error_message_setup, "ERROR", datetime.datetime.now())] if LogsView != gr.Textbox else error_message_setup
            yield {gguf_logs_output: log_entry, gguf_final_status_display: f"### ❌ Config Error\n{error_message_setup}", gguf_output_image: None}
            return

        effective_local_out_path = local_gguf_output_path or str(TEMP_DIR_ROOT / "gguf_exports_default")
        Path(effective_local_out_path).mkdir(parents=True, exist_ok=True)
        train_data_path_for_util = training_data_file.name if training_data_file and hasattr(training_data_file, 'name') else (str(training_data_file) if training_data_file else None)
        accumulated_logs_gguf = [] if LogsView != gr.Textbox else ""

        def log_callback_for_gguf_thread(raw_log_message, level_arg="INFO"):
            message_content, parsed_level = str(raw_log_message), "INFO"
            known_direct_levels = ["INFO", "DEBUG", "WARNING", "ERROR", "CRITICAL_ERROR"]
            if level_arg and level_arg.upper() in known_direct_levels: parsed_level = level_arg.upper()
            if message_content or parsed_level in ["ERROR", "CRITICAL_ERROR"]:
                log_q_gguf.put(Log(message_content, parsed_level, datetime.datetime.now()))

        gguf_thread_instance = None
        def gguf_conversion_thread_target():
            try:
                if not hasattr(gguf_utils, 'process_gguf_conversion'): raise GGUFConversionError("process_gguf_conversion missing.")
                gguf_utils.process_gguf_conversion(
                    actual_model_source_for_util,
                    effective_hf_token_for_gguf,
                    log_callback_for_gguf_thread, result_container_gguf,
                    model_id_for_util, local_model_path_for_util,
                    gguf_custom_name,
                    quantization_methods, use_importance_matrix,
                    imatrix_quant_methods, upload_to_huggingface, make_repo_private, train_data_path_for_util,
                    str(effective_local_out_path), should_split_model,
                    int(split_max_tensors_val) if split_max_tensors_val else 256, split_max_size_val,
                    str(TEMP_DIR_ROOT / "gguf_temps"), str(TEMP_DIR_ROOT / "gguf_downloads")
                )
            except Exception as e:
                result_container_gguf['error_msg'] = f"GGUF thread error: {e}"
                log_callback_for_gguf_thread(str(e), "CRITICAL_ERROR")
            finally: log_q_gguf.put(None)

        gguf_thread_instance = Thread(target=gguf_conversion_thread_target, name="GGUFConversionThread")
        gguf_thread_instance.start()
        initial_log_message_gguf = "Initiating GGUF conversion..."
        if 'LogsView' in globals() and LogsView != gr.Textbox:
            initial_log_gguf = Log(initial_log_message_gguf, "INFO", datetime.datetime.now()); accumulated_logs_gguf.append(initial_log_gguf)
            initial_log_display_gguf = accumulated_logs_gguf.copy()
        else:
            accumulated_logs_gguf = initial_log_message_gguf + "\n"; initial_log_display_gguf = accumulated_logs_gguf
        yield {gguf_logs_output: initial_log_display_gguf, gguf_final_status_display: "Starting GGUF...", gguf_output_image: None}

        while True:
            try:
                log_item_gguf = log_q_gguf.get(timeout=0.1)
                if log_item_gguf is None: break
                if 'LogsView' in globals() and LogsView != gr.Textbox:
                    accumulated_logs_gguf.append(log_item_gguf); new_logs_update_gguf = accumulated_logs_gguf.copy()
                else:
                    log_line_gguf = f"[{log_item_gguf.timestamp.strftime('%H:%M:%S')}] [{log_item_gguf.level}] {log_item_gguf.message}\n"
                    accumulated_logs_gguf += log_line_gguf; new_logs_update_gguf = accumulated_logs_gguf
                yield {gguf_logs_output: new_logs_update_gguf}
            except queue.Empty:
                if not gguf_thread_instance.is_alive(): break
                pass
        gguf_thread_instance.join(timeout=10)

        html_out, img_path, err_msg = result_container_gguf.get('final_html',""), result_container_gguf.get('image_path'), result_container_gguf.get('error_msg')
        # Use gr.update() for component updates
        status_md, img_update = "", gr.update(value=None)
        end_log_msg, end_log_lvl = ("GGUF conversion completed with errors." if err_msg else "GGUF conversion completed."), ("ERROR" if err_msg else "INFO")
        if err_msg: status_md = f"### ❌ GGUF Failed\n<pre>{str(err_msg)}</pre>"; end_log_msg = f"GGUF errors: {str(err_msg)}"
        elif html_out: status_md = html_out
        else: status_md = "GGUF finished. Check logs."
        if img_path and Path(img_path).exists():
            # Use gr.update() for component updates
            img_update = gr.update(value=img_path)
        elif img_path:
            img_warn_msg = f"Output image path {img_path} provided but image not found."
            if not ('LogsView' in globals() and LogsView != gr.Textbox): accumulated_logs_gguf += f"[{datetime.datetime.now().strftime('%H:%M:%S')}] [WARNING] {img_warn_msg}\n"
        if 'LogsView' in globals() and LogsView != gr.Textbox:
            final_log_gguf = Log(end_log_msg, end_log_lvl, datetime.datetime.now()); accumulated_logs_gguf.append(final_log_gguf)
            final_log_disp = accumulated_logs_gguf.copy()
        else:
            accumulated_logs_gguf += f"[{datetime.datetime.now().strftime('%H:%M:%S')}] [{end_log_lvl}] {end_log_msg}\n"; final_log_disp = accumulated_logs_gguf
        yield {gguf_logs_output: final_log_disp, gguf_final_status_display: status_md, gguf_output_image: img_update}

    # --- Event Handlers LINKING (after UI components are defined) ---
    demo.load(populate_examples_for_dropdown, outputs=[example_dropdown])
    load_example_btn.click(load_selected_example_content, inputs=[example_dropdown], outputs=[merge_yaml_config])

    merge_button.click(handle_merge_models,
        inputs=[
            merge_yaml_config, merge_hf_token_input, merge_repo_name,
            merge_local_save_path, merge_use_gpu, merge_use_for_gguf,
            gguf_model_source
        ],
        outputs=[
            merge_status_output, merge_logs_output, merged_model_path_state,
            gguf_hf_group, gguf_local_model_path, gguf_merged_model_display
        ])

    gguf_convert_btn.click(handle_gguf_conversion,
        inputs=[
            gguf_model_source, merged_model_path_state, gguf_model_id, gguf_local_model_path,
            gguf_custom_name_input,
            gguf_hf_token_input,
            gguf_q_methods, gguf_use_imatrix, gguf_imatrix_q_methods, gguf_train_data_file,
            gguf_upload_to_hf, gguf_private_repo, gguf_local_output_path,
            gguf_split_model, gguf_split_max_tensors, gguf_split_max_size
        ],
        outputs=[gguf_logs_output, gguf_final_status_display, gguf_output_image])

    merged_model_path_state.change(update_gguf_src_visibility,
                                   inputs=[gguf_model_source, merged_model_path_state],
                                   outputs=[gguf_hf_group, gguf_local_model_path, gguf_merged_model_display])

    gguf_model_source.change(update_gguf_src_visibility,
                             inputs=[gguf_model_source, merged_model_path_state],
                             outputs=[gguf_hf_group, gguf_local_model_path, gguf_merged_model_display])

    def initial_load_updates(merged_path, src_choice_val, imatrix_val, split_val, upload_val):
        updates = update_gguf_src_visibility(src_choice_val, merged_path)
        # Use gr.update() for component updates
        updates.update({
            gguf_q_methods: gr.update(visible=not imatrix_val),
            gguf_imatrix_q_methods: gr.update(visible=imatrix_val),
            gguf_train_data_file: gr.update(visible=imatrix_val),
            gguf_split_max_tensors: gr.update(visible=split_val),
            gguf_split_max_size: gr.update(visible=split_val),
            gguf_private_repo: gr.update(visible=upload_val)
        })
        return updates

    demo.load(initial_load_updates,
        inputs=[merged_model_path_state, gguf_model_source, gguf_use_imatrix, gguf_split_model, gguf_upload_to_hf],
        outputs=[
            gguf_hf_group, gguf_local_model_path, gguf_merged_model_display,
            gguf_q_methods, gguf_imatrix_q_methods, gguf_train_data_file,
            gguf_split_max_tensors, gguf_split_max_size, gguf_private_repo
        ])

    # Lambdas already use gr.update(), which is correct.
    gguf_use_imatrix.change(lambda u: (gr.update(visible=not u), gr.update(visible=u), gr.update(visible=u)), inputs=gguf_use_imatrix, outputs=[gguf_q_methods, gguf_imatrix_q_methods, gguf_train_data_file], api_name=False)
    gguf_split_model.change(lambda s: (gr.update(visible=s), gr.update(visible=s)), inputs=gguf_split_model, outputs=[gguf_split_max_tensors, gguf_split_max_size], api_name=False)
    gguf_upload_to_hf.change(lambda u: gr.update(visible=u), inputs=gguf_upload_to_hf, outputs=[gguf_private_repo], api_name=False)

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2, max_size=20).launch(debug=True, show_api=False)
