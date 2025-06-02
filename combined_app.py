import os
import gradio as gr
import tempfile
import shutil
from pathlib import Path
import queue
from threading import Thread
import datetime
import traceback # For detailed error logging
from typing import Optional # Added for type hinting

# --- Utility Modules ---
try:
    import gguf_utils
    if not hasattr(gguf_utils, 'GGUFConversionError'):
        class GGUFConversionError(Exception): pass
        if gguf_utils: gguf_utils.GGUFConversionError = GGUFConversionError # type: ignore
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
    if 'LogsView' not in globals(): LogsView = gr.Textbox # type: ignore # type: ignore
    if 'Log' not in globals():
        class Log: # type: ignore
            def __init__(self, message, level, timestamp):
                self.message, self.level, self.timestamp = message, level, datetime.datetime.now() if timestamp is None else timestamp


try:
    import exllamav2_utils
    if not hasattr(exllamav2_utils, 'Exllamav2Error'):
        class Exllamav2Error(Exception): pass
        if exllamav2_utils: exllamav2_utils.Exllamav2Error = Exllamav2Error # type: ignore
except ImportError:
    print("WARNING: exllamav2_utils.py not found. EXL2 quantization (Step 3) will not work.")
    exllamav2_utils = None
    class Exllamav2Error(Exception): pass


# --- Environment & Constants ---
HF_TOKEN = os.environ.get("HF_TOKEN")

APP_TEMP_ROOT_ENV_VAR = os.environ.get("APP_TEMP_ROOT")
if APP_TEMP_ROOT_ENV_VAR:
    TEMP_DIR_ROOT = Path(APP_TEMP_ROOT_ENV_VAR)
    print(f"INFO: Using temporary root directory from APP_TEMP_ROOT environment variable: {TEMP_DIR_ROOT}")
else:
    TEMP_DIR_ROOT = Path("outputs/combined_app_temp")
    print(f"INFO: APP_TEMP_ROOT environment variable not set. Using default temporary root: {TEMP_DIR_ROOT}")

TEMP_DIR_ROOT.mkdir(parents=True, exist_ok=True)
BUNDLED_IMATRIX_PATH = "/home/user/app/groups_merged.txt"

# --- Helper Functions ---
def get_hf_token_status(token, token_name):
    return f"✅ {token_name} found" if token else f"❌ {token_name} not found - set for uploads."

def extract_example_label(_, index, filename=""):
    if not filename: return f"Example {index + 1}"
    name_part = filename.removesuffix(".yaml").removesuffix(".yml").replace("_", " ").replace("-", " ")
    return ' '.join(word.capitalize() for word in name_part.split()) or f"Example {index + 1}"

def load_merge_examples():
    if not mergekit_utils:
        default_content = "models:\n  - model: cognitivecomputations/dolphin-2.6-mistral-7b-dpo-laser\n    parameters:\n      weight: 0.5\n  - model: OpenPipe/mistral-ft-optimized-1218\n    parameters:\n      weight: 0.5\nmerge_method: linear\ndtype: float16"
        return [[extract_example_label(None, 0, "Default Example.yaml"), default_content]]
    try:
        example_filenames = mergekit_utils.get_example_yaml_filenames_for_gr_examples() # type: ignore
        labeled_examples = []
        if not example_filenames or (len(example_filenames) == 1 and "default_example.yaml" in example_filenames[0]): # type: ignore
            default_content = mergekit_utils.get_example_yaml_content("default_example.yaml") # type: ignore
            labeled_examples.append([extract_example_label(None, 0, "Default Example.yaml"), default_content])
        else:
            for i, filename in enumerate(example_filenames): # type: ignore
                if filename == "default_example.yaml" and len(example_filenames) > 1: # type: ignore
                    if not any(f != "default_example.yaml" for f in example_filenames): # type: ignore
                        pass
                    else:
                        continue
                content = mergekit_utils.get_example_yaml_content(filename) # type: ignore
                if content.startswith("# Error reading"): continue
                labeled_examples.append([extract_example_label(None, i, filename), content])

        if not labeled_examples:
            default_content = mergekit_utils.get_example_yaml_content("default_example.yaml") # type: ignore
            labeled_examples.append([extract_example_label(None, 0, "Default Example.yaml"), default_content])
        return labeled_examples
    except Exception as e:
        print(f"Error loading merge examples: {e}")
        default_content = "models:\n  - model: cognitivecomputations/dolphin-2.6-mistral-7b-dpo-laser\n    parameters:\n      weight: 0.5\n  - model: OpenPipe/mistral-ft-optimized-1218\n    parameters:\n      weight: 0.5\nmerge_method: linear\ndtype: float16"
        return [[extract_example_label(None, 0, "Fallback Linear Merge.yaml"), default_content]]

def populate_examples_for_dropdown():
    examples = load_merge_examples()
    choices = [example[0] for example in examples]
    return gr.update(choices=choices, value=choices[0] if choices else None)

def load_selected_example_content(selected_label):
    if not selected_label: return ""
    examples = load_merge_examples()
    for label, content in examples:
        if label == selected_label:
            return content
    print(f"Warning: Could not find content for selected label: {selected_label}")
    return ""

def update_model_source_visibility(choice, merged_path_val, hf_group, local_path_input, merged_display):
    is_hf = (choice == "HF Hub")
    is_local = (choice == "Local Path")
    is_merged = (choice == "Output from Merge Step")
    merged_display_value = "N/A"
    if is_merged:
        if merged_path_val and Path(str(merged_path_val)).exists():
            merged_display_value = f"Using: {merged_path_val}"
        else:
            merged_display_value = "Warning: Merged path from Step 1 is invalid, not found, or Step 1 not yet run successfully."
    return { hf_group: gr.update(visible=is_hf), local_path_input: gr.update(visible=is_local), merged_display: gr.update(visible=is_merged, value=merged_display_value) }

def update_gguf_src_visibility_wrapper(choice, merged_path_val):
    return update_model_source_visibility(choice, merged_path_val, gguf_hf_group, gguf_local_model_path_input, gguf_merged_model_display)

def update_exl2_src_visibility_wrapper(choice, merged_path_val):
    return update_model_source_visibility(choice, merged_path_val, exl2_hf_group, exl2_local_model_path_input, exl2_merged_model_display)

def update_exl2_calibration_visibility(cal_mode_choice):
    return {
        exl2_cal_dataset_group: gr.update(visible=(cal_mode_choice == "Calibrate with dataset")),
        exl2_cal_measurement_file_group: gr.update(visible=(cal_mode_choice == "Use existing measurement file"))
    }

def initial_load_updates(
    merged_path_on_load, gguf_src_choice, exl2_src_choice, imatrix_val, split_val,
    merge_upload_hf_val, gguf_upload_val, exl2_upload_val_init,
    exl2_cal_mode_val, merge_passthrough_val
):
    updates = {}

    show_merged_for_gguf = merged_path_on_load and gguf_src_choice == "Output from Merge Step" and \
                           (merge_passthrough_val == "Use for GGUF & EXL2 (Steps 2 & 3)")
    show_merged_for_exl2 = merged_path_on_load and exl2_src_choice == "Output from Merge Step" and \
                           (merge_passthrough_val == "Use for GGUF & EXL2 (Steps 2 & 3)")

    updates.update(update_gguf_src_visibility_wrapper(gguf_src_choice, merged_path_on_load if show_merged_for_gguf else None))
    updates.update(update_exl2_src_visibility_wrapper(exl2_src_choice, merged_path_on_load if show_merged_for_exl2 else None))
    updates.update(update_exl2_calibration_visibility(exl2_cal_mode_val))

    updates.update({
        # Merge Tab Upload Visibility
        merge_repo_name: gr.update(visible=merge_upload_hf_val),
        merge_private_repo: gr.update(visible=merge_upload_hf_val),
        merge_hf_token_input: gr.update(visible=merge_upload_hf_val),

        # GGUF Tab Upload Visibility
        gguf_custom_name_input: gr.update(visible=gguf_upload_val),
        gguf_private_repo: gr.update(visible=gguf_upload_val),
        gguf_hf_token_input: gr.update(visible=gguf_upload_val),
        gguf_q_methods: gr.update(visible=not imatrix_val),
        gguf_imatrix_q_methods: gr.update(visible=imatrix_val),
        gguf_use_bundled_imatrix_checkbox: gr.update(visible=imatrix_val),
        gguf_split_max_tensors: gr.update(visible=split_val),
        gguf_split_max_size: gr.update(visible=split_val),

        # EXL2 Tab Upload Visibility
        exl2_custom_repo_name_base: gr.update(visible=exl2_upload_val_init),
        exl2_private_repo: gr.update(visible=exl2_upload_val_init),
        exl2_hf_token_input: gr.update(visible=exl2_upload_val_init),
    })
    return updates

# --- Gradio UI Definition ---
css = ".gradio-container {overflow-y: auto;} .checkbox-group {max-height: 200px; overflow-y: auto; border: 1px solid #e0e0e0; padding: 10px; margin: 10px 0;} .logs_view_container textarea, .Textbox textarea { font-family: monospace; font-size: 0.85em !important; white-space: pre-wrap !important; }"
with gr.Blocks(css=css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("# FusionQuant Model Merge, GGUF & EXL2 Quantization 🚀")
    merged_model_path_state = gr.State(None)
    gr.Markdown(f"**Global HF_TOKEN Status (Fallback for GGUF, Merge & EXL2):** {get_hf_token_status(HF_TOKEN, 'HF_TOKEN')}")

    with gr.Tabs():
        with gr.TabItem("Step 1: Merge Models (Mergekit)"):
            gr.Markdown("## Configure and Run Model Merge")
            if not mergekit_utils: gr.Markdown("### ❌ `mergekit_utils.py` not loaded. Merging disabled.")
            with gr.Row():
                with gr.Column(scale=2):
                    merge_yaml_config = gr.Code(label="Mergekit YAML", language="yaml", lines=40)
                    if mergekit_utils and hasattr(mergekit_utils, 'create_example_files'):
                        try: mergekit_utils.create_example_files() # type: ignore
                        except Exception as e: print(f"Error creating merge examples: {e}")

                    gr.Markdown("### Load Merge Example")
                    with gr.Row():
                        example_dropdown = gr.Dropdown(choices=[], label="Select Example", interactive=True)
                        load_example_btn = gr.Button("Load Example", size="sm")
                with gr.Column(scale=1):
                    gr.Markdown("#### Merge Output & Upload Settings")
                    merge_upload_to_hf = gr.Checkbox(label="Upload merged model to HF", value=False)
                    merge_private_repo = gr.Checkbox(label="Make Merge Repo Private", value=False, visible=True)
                    merge_repo_name = gr.Textbox(label="Custom HF Repo Name (Merge Upload)", placeholder="e.g., MyModel", visible=True)
                    merge_hf_token_input = gr.Textbox(label="HF Token (Merge upload)", type="password", placeholder="Uses global HF_TOKEN env var if blank.", visible=True)
                    merge_local_save_path = gr.Textbox(label="Local Save Path", placeholder=f"e.g., {TEMP_DIR_ROOT}/my-merged-model")
                    merge_keep_hf_cache_checkbox = gr.Checkbox(label="Keep downloaded Hugging Face models locally (persistent cache)", value=False, info="If checked, provide a path below. Otherwise, downloads are temporary.")
                    merge_persistent_hf_cache_path_input = gr.Textbox(label="Persistent HF Cache Path", placeholder="e.g., /path/to/my/hf_models_cache", visible=False, info="Models downloaded for merging will be stored here.")

                    gr.Markdown("#### Merge Settings")
                    merge_use_gpu = gr.Checkbox(label="Use GPU for Merge (if available)", value=True, info="Uncheck to force CPU merge.")
                    merge_passthrough_option = gr.Radio(
                        ["Do not automatically use", "Use for GGUF & EXL2 (Steps 2 & 3)"],
                        label="Use merged model for next steps:",
                        value="Do not automatically use"
                    )
            # Order: Button, Status, Logs (This tab was already correct)
            merge_button = gr.Button("Run Merge", variant="primary", interactive=mergekit_utils is not None)
            merge_status_output = gr.Markdown()
            log_elem_class = "logs_view_container" if LogsView != gr.Textbox else "Textbox"
            if 'LogsView' in globals() and LogsView != gr.Textbox:
                merge_logs_output = LogsView(label="Merge Logs", lines=15, elem_classes=log_elem_class)
            else:
                merge_logs_output = gr.Textbox(label="Merge Logs", lines=15, interactive=False, elem_classes=log_elem_class)


        with gr.TabItem("Step 2: Convert to GGUF & Quantize (Llama.cpp)"):
            gr.Markdown("## Configure and Run GGUF Quantization")
            if not gguf_utils: gr.Markdown("### ❌ `gguf_utils.py` not loaded. GGUF disabled.")
            gguf_model_source = gr.Radio(["Output from Merge Step", "HF Hub", "Local Path"], label="GGUF Model Source", value="Output from Merge Step")
            with gr.Group(visible=False) as gguf_hf_group:
                gguf_model_id = gr.Textbox(label="HF Model ID", placeholder="e.g., meta-llama/Llama-2-7b-hf")
            gguf_local_model_path_input = gr.Textbox(label="Local Model Path", placeholder="/path/to/model_dir", visible=False)
            gguf_merged_model_display = gr.Textbox(label="Using Merged Model Path:", interactive=False, visible=True)
            
            gr.Markdown("### GGUF Output & Upload Settings")
            gguf_upload_to_hf = gr.Checkbox(label="Upload GGUF model(s) to HF", value=False)     
            gguf_private_repo = gr.Checkbox(label="Make GGUF Repo Private", value=False, visible=True)
            gguf_custom_name_input = gr.Textbox(label="Custom HF Repo Name (GGUF Upload)", placeholder="e.g., MyModel (GGUF suffix auto-added)", visible=True)
            gguf_hf_token_input = gr.Textbox(label="HF Token (GGUF Upload)", type="password", placeholder="Uses global HF_TOKEN env var if blank.", visible=True)
            gguf_local_output_path = gr.Textbox(label="Local Save Path", placeholder=f"e.g., {TEMP_DIR_ROOT}/gguf_exports (quant specific name auto-added)")

            gr.Markdown("### Quantization Settings")
            gguf_q_methods = gr.CheckboxGroup(["Q2_K", "Q2_K_L", "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q4_0", "Q4_K_S", "Q4_K_M", "Q5_0", "Q5_K_S", "Q5_K_M", "Q5_K_L", "Q6_K", "Q8_0", "FP16"], label="Standard Quants", value=["Q5_K_M"], elem_classes="checkbox-group")
            gguf_use_imatrix = gr.Checkbox(label="Use Importance Matrix", value=False)
            gguf_imatrix_q_methods = gr.CheckboxGroup(["IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M","IQ3_XXS", "IQ3_XS", "IQ3_XXS", "IQ3_M", "Q4_K_M", "Q4_K_S", "IQ4_NL", "IQ4_XS", "Q5_K_M", "Q5_K_S"], label="Imatrix Quants", value=["IQ4_XS"], visible=False, elem_classes="checkbox-group")
            gguf_use_bundled_imatrix_checkbox = gr.Checkbox(label=f"Use bundled groups_merged.txt for Imatrix (Path: {BUNDLED_IMATRIX_PATH})", value=False, visible=False, info=f"If checked and 'Use Importance Matrix' is active, {BUNDLED_IMATRIX_PATH} will be used." )
            
            gr.Markdown("### Sharding Settings")
            gguf_split_model = gr.Checkbox(label="Split GGUF Model Shards")
            gguf_split_max_tensors = gr.Number(label="Max Tensors/Shard", value=256, visible=False)
            gguf_split_max_size = gr.Textbox(label="Max Size/Shard (e.g., 5G)", visible=False)
            
            # MODIFIED ORDER: Button, Status, Logs
            gguf_convert_btn = gr.Button("Convert to GGUF & Quantize", variant="primary", interactive=gguf_utils is not None)
            gguf_final_status_display = gr.Markdown() # Moved UP
            log_elem_class_gguf = "logs_view_container" if LogsView != gr.Textbox else "Textbox"
            if 'LogsView' in globals() and LogsView != gr.Textbox:
                gguf_logs_output = LogsView(label="GGUF Quantization Logs", lines=15, elem_classes=log_elem_class_gguf)
            else:
                gguf_logs_output = gr.Textbox(label="GGUF Quantization Logs", lines=15, interactive=False, elem_classes=log_elem_class_gguf)


        with gr.TabItem("Step 3: Quantize to EXL2 (Exllamav2)"):
            gr.Markdown("## Configure and Run EXL2 Quantization")
            if not exllamav2_utils: gr.Markdown("### ❌ `exllamav2_utils.py` not loaded. EXL2 quantization disabled.")
            exl2_model_source = gr.Radio(["Output from Merge Step", "HF Hub", "Local Path"], label="EXL2 Model Source", value="Output from Merge Step")
            with gr.Group(visible=False) as exl2_hf_group:
                exl2_hf_model_id = gr.Textbox(label="HF Model ID", placeholder="e.g., meta-llama/Llama-2-7b-hf")
            exl2_local_model_path_input = gr.Textbox(label="Local Model Path", placeholder="/path/to/model_dir", visible=False)
            exl2_merged_model_display = gr.Textbox(label="Using Merged Model Path:", interactive=False, visible=True)

            gr.Markdown("### EXL2 Output & Upload Settings")
            exl2_upload_to_hf = gr.Checkbox(label="Upload EXL2 Model(s) to HF", value=False)
            exl2_private_repo = gr.Checkbox(label="Make EXL2 Repo(s) Private", value=False, visible=True)
            exl2_custom_repo_name_base = gr.Textbox(label="Custom HF Repo Name (EXL2 Upload)", placeholder="e.g., MyModel (EXL2+BPW+Headbit suffix auto-added)", visible=True)
            exl2_hf_token_input = gr.Textbox(label="HF Token (EXL2 Upload)", type="password", placeholder="Uses global HF_TOKEN env var if blank.", visible=True)
            exl2_local_output_dir_base = gr.Textbox(label="Local Save Path", placeholder=f"e.g., {TEMP_DIR_ROOT}/exl2_exports/my_model_base (quant specific name auto-added)")
            
            gr.Markdown("### EXL2 Quantization Settings")
            exl2_bits_input_str = gr.Textbox(label="Bits per Weight (Comma-separated list, e.g., 4.0, 4.58, 6.0)", value="4.58", info="Enter one or more target bitrates. If multiple are chosen measurement is auto-reused.")
            exl2_head_bits = gr.Number(label="Head Bits (Optional)", minimum=0, maximum=8, step=1, value=0, info="Bitrate for final layers. 0 to use main bits or disable.")
            
            gr.Markdown("### Calibration Settings")
            exl2_calibration_mode = gr.Radio(["No explicit calibration", "Calibrate with dataset", "Use existing measurement file"], label="Calibration Mode", value="No explicit calibration" )
            with gr.Group(visible=True) as exl2_cal_dataset_group: # Default for "No explicit calibration" means this group should be visible if that's default
                exl2_cal_dataset = gr.Textbox(label="Calibration Dataset (HF name or local .parquet)", placeholder="e.g., wikitext-2-raw-v1 or /path/to/data.parquet", value="wikitext-2-raw-v1")
                exl2_cal_rows = gr.Number(label="Calibration Rows", value=100, minimum=10, step=10)
            with gr.Group(visible=False) as exl2_cal_measurement_file_group:
                exl2_input_measurement_file = gr.Textbox(label="Path to measurement.json file (Overrides dataset cal.)", placeholder="/path/to/measurement.json")
            
            # MODIFIED ORDER: Button, Status, Logs
            exl2_quantize_btn = gr.Button("Convert to EXL2 & Quantize", variant="primary", interactive=exllamav2_utils is not None)
            exl2_final_status_display = gr.Markdown() # Moved UP
            log_elem_class_exl2 = "logs_view_container" if LogsView != gr.Textbox else "Textbox"
            if 'LogsView' in globals() and LogsView != gr.Textbox:
                exl2_logs_output = LogsView(label="EXL2 Quantization Logs", lines=15, elem_classes=log_elem_class_exl2)
            else:
                exl2_logs_output = gr.Textbox(label="EXL2 Quantization Logs", lines=15, interactive=False, elem_classes=log_elem_class_exl2)


    # --- Main Handler Function Definitions ---
    def handle_merge_models(
        yaml_config,
        merge_upload_to_hf_val,
        hf_token_merge_input, repo_name_merge, merge_repo_private_val,
        local_save_path,
        keep_hf_cache_for_merge, persistent_hf_cache_path_for_merge,
        use_gpu_for_merge, passthrough_choice,
        current_gguf_model_source_choice, current_exl2_model_source_choice
    ):
        if not mergekit_utils:
            error_msg = "Mergekit utils not loaded."
            log_entry = [Log(error_msg, "ERROR", None)] if LogsView != gr.Textbox else error_msg # type: ignore
            updates = {merge_status_output: error_msg, merge_logs_output: log_entry, merged_model_path_state: None}
            updates.update(update_gguf_src_visibility_wrapper(current_gguf_model_source_choice, None)) # type: ignore
            updates.update(update_exl2_src_visibility_wrapper(current_exl2_model_source_choice, None)) # type: ignore
            yield updates
            return

        effective_hf_token_for_merge = hf_token_merge_input or HF_TOKEN
        output_path_for_mergekit_process = local_save_path
        persistent_passthrough_dir_for_next_step = None
        
        effective_repo_name_merge = repo_name_merge if merge_upload_to_hf_val else None

        use_for_gguf_exl2 = passthrough_choice == "Use for GGUF & EXL2 (Steps 2 & 3)"
        use_for_any_next_step = use_for_gguf_exl2

        if use_for_any_next_step and not local_save_path:
            dir_name = f"merged_passthrough_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            persistent_passthrough_dir_for_next_step = TEMP_DIR_ROOT / dir_name
            persistent_passthrough_dir_for_next_step.mkdir(parents=True, exist_ok=True)
            output_path_for_mergekit_process = str(persistent_passthrough_dir_for_next_step)
        elif local_save_path:
            Path(local_save_path).parent.mkdir(parents=True, exist_ok=True)

        if not output_path_for_mergekit_process and not effective_repo_name_merge:
            error_msg = "No local save path, HF repository name for upload (with upload enabled), or passthrough option selected."
            log_entry = [Log(error_msg, "ERROR", None)] if LogsView != gr.Textbox else error_msg # type: ignore
            updates = {merge_status_output: f"### ❌ Config Error\n{error_msg}", merge_logs_output: log_entry, merged_model_path_state: None}
            updates.update(update_gguf_src_visibility_wrapper(current_gguf_model_source_choice, None)) # type: ignore
            updates.update(update_exl2_src_visibility_wrapper(current_exl2_model_source_choice, None)) # type: ignore
            yield updates
            return

        community_token = getattr(mergekit_utils, 'community_hf_token_val', None) if mergekit_utils else None
        if effective_repo_name_merge and not effective_hf_token_for_merge and not (effective_repo_name_merge.startswith("mergekit-community/") and community_token):
            error_msg = f"HF Repo '{effective_repo_name_merge}' specified for merge, but no effective HF token."
            log_entry = [Log(error_msg, "ERROR", None)] if LogsView != gr.Textbox else error_msg # type: ignore
            updates = {merge_status_output: f"### ❌ Merge Error\n{error_msg}", merge_logs_output: log_entry, merged_model_path_state: None}
            updates.update(update_gguf_src_visibility_wrapper(current_gguf_model_source_choice, None)) # type: ignore
            updates.update(update_exl2_src_visibility_wrapper(current_exl2_model_source_choice, None)) # type: ignore
            yield updates
            return

        log_q, result_container = queue.Queue(), {}
        accumulated_logs = [] if LogsView != gr.Textbox else "" # type: ignore

        def log_callback_for_thread(raw_log_message, level_arg="INFO"):
            message_content = str(raw_log_message); parsed_level = "INFO"
            prefix_map = [("[STDOUT]", "INFO"), ("[STDERR]", "ERROR"), ("[DEBUG]", "DEBUG"), ("[INFO]", "INFO"), ("[WARNING]", "WARNING"), ("[ERROR]", "ERROR")]
            found_prefix = False
            for p_str, p_lvl in prefix_map:
                if message_content.startswith(p_str): parsed_level = p_lvl; message_content = message_content[len(p_str):].lstrip(); found_prefix = True; break
            if not found_prefix and level_arg: parsed_level = level_arg.upper()
            if message_content or parsed_level == "ERROR": log_q.put(Log(message_content, parsed_level, datetime.datetime.now())) # type: ignore

        merge_thread_instance = None
        def merge_thread_target():
            try:
                _, final_path_from_util, error_msg_from_util = mergekit_utils.process_model_merge( # type: ignore
                    yaml_config_str=yaml_config, hf_token_merge=effective_hf_token_for_merge,
                    repo_name=effective_repo_name_merge, 
                    hf_repo_private_for_merge=merge_repo_private_val,
                    local_path_merge_output=output_path_for_mergekit_process,
                    community_hf_token_val=community_token, use_gpu_bool=use_gpu_for_merge,
                    temp_dir_base=str(TEMP_DIR_ROOT / "mergekit_job_temps"),
                    log_fn=log_callback_for_thread,
                    keep_hf_cache=keep_hf_cache_for_merge,
                    persistent_hf_cache_path=persistent_hf_cache_path_for_merge
                )
                result_container['final_path'] = final_path_from_util
                result_container['error_msg'] = error_msg_from_util
            except Exception as e_thread:
                err_msg_thread = f"Critical error in merge thread: {str(e_thread)}\n{traceback.format_exc()}"
                result_container['error_msg'] = err_msg_thread
                log_callback_for_thread(err_msg_thread, "CRITICAL_ERROR")
            finally:
                log_q.put(None)

        merge_thread_instance = Thread(target=merge_thread_target, name="MergeThread")
        merge_thread_instance.start()
        initial_log_message = "Initiating merge process..."
        if 'LogsView' in globals() and LogsView != gr.Textbox: # type: ignore
            initial_log = Log(initial_log_message, "INFO", datetime.datetime.now()) # type: ignore
            accumulated_logs.append(initial_log); initial_log_display = accumulated_logs.copy() # type: ignore
        else:
            accumulated_logs = initial_log_message + "\n"; initial_log_display = accumulated_logs # type: ignore
        yield {merge_status_output: "Starting merge...", merge_logs_output: initial_log_display, merged_model_path_state: None}

        while True:
            try:
                log_item = log_q.get(timeout=0.1)
                if log_item is None: break
                if 'LogsView' in globals() and LogsView != gr.Textbox: # type: ignore
                    accumulated_logs.append(log_item); new_logs_update = accumulated_logs.copy() # type: ignore
                else:
                    log_line = f"[{log_item.timestamp.strftime('%H:%M:%S')}] [{log_item.level}] {log_item.message}\n" # type: ignore
                    accumulated_logs += log_line; new_logs_update = accumulated_logs # type: ignore
                yield {merge_logs_output: new_logs_update}
            except queue.Empty:
                if not merge_thread_instance.is_alive(): break; pass
        merge_thread_instance.join(timeout=10)

        final_path, err_msg = result_container.get('final_path'), result_container.get('error_msg')
        status_md, next_step_path_state = "", None
        if not err_msg and final_path:
            status_md = f"### ✅ Merge Successful!\nOutput: `{final_path}`"
            if use_for_any_next_step:
                next_step_path_state = final_path
                status_md += f"\nPath set for GGUF & EXL2 (Steps 2 & 3)."
        elif err_msg:
            status_md = f"### ❌ Merge Failed\n<pre>{err_msg}</pre>"
            if persistent_passthrough_dir_for_next_step and persistent_passthrough_dir_for_next_step.exists():
                try: shutil.rmtree(persistent_passthrough_dir_for_next_step)
                except Exception as e_clean: log_callback_for_thread(f"Error cleaning passthrough dir: {e_clean}", "WARNING")
        else: status_md = "Merge outcome unclear. Check logs."

        end_log_msg = (f"Merge finished with errors: {err_msg}" if err_msg else "Merge process finished.")
        end_log_lvl = ("ERROR" if err_msg else "INFO")
        if 'LogsView' in globals() and LogsView != gr.Textbox: # type: ignore
            final_log = Log(end_log_msg, end_log_lvl, datetime.datetime.now()) # type: ignore
            accumulated_logs.append(final_log); final_log_disp = accumulated_logs.copy() # type: ignore
        else:
            accumulated_logs += f"[{datetime.datetime.now().strftime('%H:%M:%S')}] [{end_log_lvl}] {end_log_msg}\n"; final_log_disp = accumulated_logs # type: ignore

        final_yield_updates = {
            merge_status_output: status_md, merge_logs_output: final_log_disp,
            merged_model_path_state: next_step_path_state }
        final_yield_updates.update(update_gguf_src_visibility_wrapper(current_gguf_model_source_choice, next_step_path_state if use_for_gguf_exl2 else None)) # type: ignore
        final_yield_updates.update(update_exl2_src_visibility_wrapper(current_exl2_model_source_choice, next_step_path_state if use_for_gguf_exl2 else None)) # type: ignore
        yield final_yield_updates


    def handle_gguf_conversion(
        model_source_choice, current_merged_model_path, hf_model_id_input, local_model_path_val,
        gguf_upload_to_hf_val, gguf_custom_name_val, gguf_private_repo_val, gguf_hf_token_val,
        local_gguf_output_path_val,
        quantization_methods_selected, use_importance_matrix, imatrix_quant_methods_selected,
        use_bundled_imatrix_checkbox_value,
        should_split_model, split_max_tensors_val, split_max_size_val
    ):
        if not gguf_utils:
            error_msg = "GGUF utils (gguf_utils.py) not loaded."
            log_entry = [Log(error_msg, "ERROR", None)] if LogsView != gr.Textbox else error_msg # type: ignore
            yield {gguf_logs_output: log_entry, gguf_final_status_display: f"### ❌ Error\n{error_msg}"}
            return

        effective_hf_token_for_gguf = gguf_hf_token_val or HF_TOKEN
        log_q_gguf, result_container_gguf = queue.Queue(), {}
        actual_model_source_for_util: Optional[str] = None
        model_id_for_util: Optional[str] = None
        local_model_path_for_util: Optional[str] = None
        error_message_setup: Optional[str] = None
        original_model_name_part_for_default: str = "default_gguf_model_name"

        if model_source_choice == "Output from Merge Step":
            if not current_merged_model_path or not Path(str(current_merged_model_path)).exists():
                error_message_setup = "Merged model path from Step 1 is invalid or does not exist."
            else:
                actual_model_source_for_util, local_model_path_for_util = "Local Path", str(current_merged_model_path)
                original_model_name_part_for_default = Path(str(current_merged_model_path)).name
        elif model_source_choice == "HF Hub":
            if not hf_model_id_input: error_message_setup = "Hugging Face Model ID is required."
            else:
                actual_model_source_for_util, model_id_for_util = "HF Hub", hf_model_id_input
                original_model_name_part_for_default = Path(hf_model_id_input).name
        elif model_source_choice == "Local Path":
            if not local_model_path_val or not Path(local_model_path_val).exists():
                error_message_setup = "Specified local model path is invalid or does not exist."
            else:
                actual_model_source_for_util, local_model_path_for_util = "Local Path", str(local_model_path_val)
                original_model_name_part_for_default = Path(local_model_path_val).name
        else:
            error_message_setup = "Invalid GGUF model source selected."

        if error_message_setup:
            log_entry = [Log(error_message_setup, "ERROR", None)] if LogsView != gr.Textbox else error_message_setup # type: ignore
            yield {gguf_logs_output: log_entry, gguf_final_status_display: f"### ❌ Config Error\n{error_message_setup}"}
            return

        user_did_specify_gguf_save_path = bool(local_gguf_output_path_val and local_gguf_output_path_val.strip())
        effective_local_out_path = local_gguf_output_path_val

        if not user_did_specify_gguf_save_path:
            default_folder_base_name = gguf_custom_name_val or original_model_name_part_for_default
            effective_local_out_path = str(TEMP_DIR_ROOT / "gguf_exports_default" / default_folder_base_name)

        Path(effective_local_out_path).mkdir(parents=True, exist_ok=True) # type: ignore

        train_data_path_for_util = None
        if use_importance_matrix:
            if use_bundled_imatrix_checkbox_value:
                if Path(BUNDLED_IMATRIX_PATH).is_file(): train_data_path_for_util = BUNDLED_IMATRIX_PATH
                else: print(f"WARNING: Bundled imatrix file '{BUNDLED_IMATRIX_PATH}' selected but not found.")

        accumulated_logs_gguf = [] if LogsView != gr.Textbox else "" # type: ignore

        def log_callback_for_gguf_thread(raw_log_message, level_arg="INFO"):
            message_content, parsed_level = str(raw_log_message), "INFO"
            known_direct_levels = ["INFO", "DEBUG", "WARNING", "ERROR", "CRITICAL_ERROR", "STDOUT"]
            if level_arg and level_arg.upper() in known_direct_levels:
                parsed_level = level_arg.upper()
                if parsed_level == "STDOUT": parsed_level = "INFO"
            if message_content or parsed_level in ["ERROR", "CRITICAL_ERROR"]:
                log_q_gguf.put(Log(message_content, parsed_level, datetime.datetime.now())) # type: ignore

        if use_importance_matrix:
            if use_bundled_imatrix_checkbox_value and train_data_path_for_util:
                 log_callback_for_gguf_thread(f"Confirmed: Using bundled imatrix file: {train_data_path_for_util}", "INFO")
            elif use_bundled_imatrix_checkbox_value and not train_data_path_for_util:
                 log_callback_for_gguf_thread(f"WARNING: Bundled imatrix file selected but not found.", "WARNING")
            elif train_data_path_for_util is None:
                 log_callback_for_gguf_thread("INFO: Importance matrix enabled, no specific training data provided.", "INFO")

        gguf_thread_instance = None
        def gguf_conversion_thread_target():
            try:
                if not hasattr(gguf_utils, 'process_gguf_conversion'):
                    raise GGUFConversionError("process_gguf_conversion function is missing from gguf_utils.")

                gguf_utils.process_gguf_conversion( # type: ignore
                    model_source_type=str(actual_model_source_for_util),
                    hf_token=effective_hf_token_for_gguf,
                    log_fn=log_callback_for_gguf_thread,
                    result_container=result_container_gguf,
                    hf_model_id=model_id_for_util,
                    local_model_path=local_model_path_for_util,
                    custom_model_name_gguf=gguf_custom_name_val if gguf_upload_to_hf_val else None,
                    quant_methods_list=quantization_methods_selected,
                    use_imatrix_bool=use_importance_matrix,
                    imatrix_quant_methods_list=imatrix_quant_methods_selected,
                    upload_to_hf_bool=gguf_upload_to_hf_val,
                    hf_repo_private_bool=gguf_private_repo_val,
                    train_data_path_gguf=train_data_path_for_util,
                    output_dir_gguf=str(effective_local_out_path),
                    user_specified_gguf_save_path=user_did_specify_gguf_save_path,
                    split_model_bool=should_split_model,
                    split_max_tensors_val=int(split_max_tensors_val) if split_max_tensors_val is not None else 256,
                    split_max_size_val=split_max_size_val,
                    temp_dir_base_for_job=str(TEMP_DIR_ROOT / "gguf_temps"),
                    download_dir_for_job=str(TEMP_DIR_ROOT / "gguf_downloads")
                )
            except Exception as e_gguf_thread:
                tb_str_exc_gguf = traceback.format_exc()
                result_container_gguf['error_msg'] = f"GGUF Quantization thread error: {str(e_gguf_thread)}"
                log_callback_for_gguf_thread(f"Critical GGUF Thread Error: {str(e_gguf_thread)}\nTraceback:\n{tb_str_exc_gguf}", "CRITICAL_ERROR")
            finally:
                log_q_gguf.put(None)

        gguf_thread_instance = Thread(target=gguf_conversion_thread_target, name="GGUFConversionThread")
        gguf_thread_instance.start()
        initial_log_message_gguf = "Initiating GGUF Quantization..."
        if 'LogsView' in globals() and LogsView != gr.Textbox: # type: ignore
            initial_log_gguf = Log(initial_log_message_gguf, "INFO", datetime.datetime.now()) # type: ignore
            accumulated_logs_gguf.append(initial_log_gguf); initial_log_display_gguf = accumulated_logs_gguf.copy() # type: ignore
        else:
            accumulated_logs_gguf = initial_log_message_gguf + "\n" ; initial_log_display_gguf = accumulated_logs_gguf # type: ignore
        yield {gguf_logs_output: initial_log_display_gguf, gguf_final_status_display: "Starting GGUF Quantization..."}

        while True:
            try:
                log_item_gguf = log_q_gguf.get(timeout=0.1)
                if log_item_gguf is None: break
                if 'LogsView' in globals() and LogsView != gr.Textbox: # type: ignore
                    accumulated_logs_gguf.append(log_item_gguf); new_logs_update_gguf = accumulated_logs_gguf.copy() # type: ignore
                else:
                    log_line_gguf = f"[{log_item_gguf.timestamp.strftime('%H:%M:%S')}] [{log_item_gguf.level}] {log_item_gguf.message}\n" # type: ignore
                    accumulated_logs_gguf += log_line_gguf; new_logs_update_gguf = accumulated_logs_gguf # type: ignore
                yield {gguf_logs_output: new_logs_update_gguf}
            except queue.Empty:
                if not gguf_thread_instance.is_alive(): break; pass
        gguf_thread_instance.join(timeout=10)

        final_status_md_parts = []
        if 'final_status_messages_list' in result_container_gguf and result_container_gguf['final_status_messages_list']:
            final_status_md_parts.append("### GGUF Quantization Summary:")
            for msg_item in result_container_gguf['final_status_messages_list']:
                final_status_md_parts.append(f"- {msg_item}")
            status_md = "\n".join(final_status_md_parts)
        else:
             status_md = "GGUF process finished. Check logs for details."

        err_msg = result_container_gguf.get('error_msg')
        end_log_msg = "GGUF Quantization process completed."
        end_log_lvl = "INFO"
        if err_msg:
            end_log_msg = f"GGUF Quantization finished with errors: {str(err_msg)}"
            end_log_lvl = "ERROR"
            if not (result_container_gguf.get('final_status_messages_list') and any(s.startswith("✅") for s in result_container_gguf['final_status_messages_list'])):
                status_md = f"### ❌ GGUF Quantization Failed\nOverall Error: {str(err_msg)}"

        if 'LogsView' in globals() and LogsView != gr.Textbox: # type: ignore
            final_log_gguf = Log(end_log_msg, end_log_lvl, datetime.datetime.now()) # type: ignore
            accumulated_logs_gguf.append(final_log_gguf); final_log_disp_gguf = accumulated_logs_gguf.copy() # type: ignore
        else:
            accumulated_logs_gguf += f"[{datetime.datetime.now().strftime('%H:%M:%S')}] [{end_log_lvl}] {end_log_msg}\n"; final_log_disp_gguf = accumulated_logs_gguf # type: ignore

        yield {gguf_logs_output: final_log_disp_gguf, gguf_final_status_display: status_md}


    def handle_exl2_quantization(
        model_source_choice_exl2, current_merged_model_path_val,
        hf_model_id_exl2_input, local_model_path_exl2_val,
        exl2_upload_to_hf_val, exl2_custom_repo_name_base_val, exl2_private_repo_val, hf_token_exl2_val,
        local_output_dir_base_exl2_val,
        bits_input_str_exl2,
        head_bits_exl2_input,
        calibration_mode_exl2, cal_dataset_exl2_input, cal_rows_exl2_input, input_measurement_file_exl2_input
    ):
        if not exllamav2_utils:
            error_msg = "Exllamav2 utils (exllamav2_utils.py) not loaded."
            log_entry = [Log(error_msg, "ERROR", None)] if LogsView != gr.Textbox else error_msg # type: ignore
            yield {exl2_logs_output: log_entry, exl2_final_status_display: f"### ❌ Error\n{error_msg}"}
            return

        selected_bits_exl2_list = []
        if bits_input_str_exl2 and bits_input_str_exl2.strip():
            try:
                selected_bits_exl2_list = [float(b.strip()) for b in bits_input_str_exl2.split(',') if b.strip()]
            except ValueError:
                error_msg = "Invalid format for EXL2 Bits per Weight."
                log_entry = [Log(error_msg, "ERROR", None)] if LogsView != gr.Textbox else error_msg # type: ignore
                yield {exl2_logs_output: log_entry, exl2_final_status_display: f"### ❌ Config Error\n{error_msg}"}
                return
        if not selected_bits_exl2_list:
            error_msg = "No EXL2 bitrates entered."
            log_entry = [Log(error_msg, "ERROR", None)] if LogsView != gr.Textbox else error_msg # type: ignore
            yield {exl2_logs_output: log_entry, exl2_final_status_display: f"### ❌ Config Error\n{error_msg}"}
            return

        log_q_exl2, result_container_exl2 = queue.Queue(), {}
        actual_model_input_for_util: Optional[str] = None
        error_message_setup_exl2: Optional[str] = None
        original_model_name_part_exl2: str = "default_exl2_model_name"

        if model_source_choice_exl2 == "Output from Merge Step":
            if not current_merged_model_path_val or not Path(str(current_merged_model_path_val)).exists():
                error_message_setup_exl2 = "Merged model path from Step 1 is invalid."
            else:
                actual_model_input_for_util = str(current_merged_model_path_val)
                original_model_name_part_exl2 = Path(str(current_merged_model_path_val)).name
        elif model_source_choice_exl2 == "HF Hub":
            if not hf_model_id_exl2_input: error_message_setup_exl2 = "HF Model ID is required for EXL2."
            else:
                actual_model_input_for_util = hf_model_id_exl2_input
                original_model_name_part_exl2 = Path(hf_model_id_exl2_input).name
        elif model_source_choice_exl2 == "Local Path":
            if not local_model_path_exl2_val or not Path(local_model_path_exl2_val).exists():
                error_message_setup_exl2 = "Specified local model path for EXL2 is invalid."
            else:
                actual_model_input_for_util = str(local_model_path_exl2_val)
                original_model_name_part_exl2 = Path(local_model_path_exl2_val).name
        else:
            error_message_setup_exl2 = "Invalid EXL2 model source."

        if error_message_setup_exl2:
            log_entry = [Log(error_message_setup_exl2, "ERROR", None)] if LogsView != gr.Textbox else error_message_setup_exl2 # type: ignore
            yield {exl2_logs_output: log_entry, exl2_final_status_display: f"### ❌ Config Error\n{error_message_setup_exl2}"}
            return

        user_did_specify_local_path_exl2 = bool(local_output_dir_base_exl2_val and local_output_dir_base_exl2_val.strip())
        effective_local_exl2_out_dir_base = local_output_dir_base_exl2_val

        if not user_did_specify_local_path_exl2:
            default_folder_base_name_exl2 = exl2_custom_repo_name_base_val or original_model_name_part_exl2
            effective_local_exl2_out_dir_base = str(TEMP_DIR_ROOT / "exl2_exports_default" / default_folder_base_name_exl2)

        accumulated_logs_exl2 = [] if LogsView != gr.Textbox else "" # type: ignore

        def log_callback_for_exl2_thread(raw_log_message, level_arg="INFO"):
            message_content, parsed_level = str(raw_log_message), "INFO"
            if level_arg and level_arg.upper() in ["INFO", "DEBUG", "WARNING", "ERROR", "CRITICAL_ERROR"]:
                parsed_level = level_arg.upper()
            if message_content or parsed_level in ["ERROR", "CRITICAL_ERROR"]:
                log_q_exl2.put(Log(message_content, parsed_level, datetime.datetime.now())) # type: ignore

        exl2_thread_instance = None
        def exl2_quant_thread_target():
            try:
                if not hasattr(exllamav2_utils, 'process_exllamav2_quantization'):
                    raise Exllamav2Error("process_exllamav2_quantization function is missing from exllamav2_utils.")
                exllamav2_utils.process_exllamav2_quantization( # type: ignore
                    model_source_type=model_source_choice_exl2,
                    hf_model_id_or_path=str(actual_model_input_for_util),
                    output_exl2_model_dir_base=str(effective_local_exl2_out_dir_base),
                    bits_list=selected_bits_exl2_list,
                    head_bits=int(head_bits_exl2_input) if head_bits_exl2_input is not None and float(head_bits_exl2_input) > 0 else 0, # type: ignore
                    log_fn=log_callback_for_exl2_thread,
                    result_container=result_container_exl2,
                    calibration_mode=calibration_mode_exl2.lower().replace(" ", "_").replace("-","_"),
                    calibration_dataset_path_or_hf_name=cal_dataset_exl2_input,
                    calibration_rows=int(cal_rows_exl2_input) if cal_rows_exl2_input is not None else 100, # type: ignore
                    input_measurement_file_path=input_measurement_file_exl2_input,
                    hf_token_exl2=(hf_token_exl2_val or HF_TOKEN),
                    upload_to_hf_exl2=exl2_upload_to_hf_val,
                    hf_repo_name_base_exl2=exl2_custom_repo_name_base_val if exl2_upload_to_hf_val else None,
                    hf_repo_private_exl2=exl2_private_repo_val,
                    user_specified_local_output_path=user_did_specify_local_path_exl2,
                    temp_dir_base=str(TEMP_DIR_ROOT / "exl2_temps")
                )
            except Exception as e_exl2_thread:
                tb_str_exc_exl2 = traceback.format_exc()
                result_container_exl2['error_message'] = f"EXL2 quantization thread error: {str(e_exl2_thread)}"
                log_callback_for_exl2_thread(f"Critical EXL2 Thread Error: {str(e_exl2_thread)}\nTraceback:\n{tb_str_exc_exl2}", "CRITICAL_ERROR")
            finally:
                log_q_exl2.put(None)

        exl2_thread_instance = Thread(target=exl2_quant_thread_target, name="EXL2QuantThread")
        exl2_thread_instance.start()
        initial_log_msg_exl2 = "Initiating EXL2 quantization..."
        if 'LogsView' in globals() and LogsView != gr.Textbox: # type: ignore
            initial_log_exl2 = Log(initial_log_msg_exl2, "INFO", datetime.datetime.now()) # type: ignore
            accumulated_logs_exl2.append(initial_log_exl2); initial_log_display_exl2 = accumulated_logs_exl2.copy() # type: ignore
        else:
            accumulated_logs_exl2 = initial_log_msg_exl2 + "\n"; initial_log_display_exl2 = accumulated_logs_exl2 # type: ignore
        yield {exl2_logs_output: initial_log_display_exl2, exl2_final_status_display: "Starting EXL2 quantization..."}

        while True:
            try:
                log_item_exl2 = log_q_exl2.get(timeout=0.1)
                if log_item_exl2 is None: break
                if 'LogsView' in globals() and LogsView != gr.Textbox: # type: ignore
                    accumulated_logs_exl2.append(log_item_exl2); new_logs_update_exl2 = accumulated_logs_exl2.copy() # type: ignore
                else:
                    log_line_exl2 = f"[{log_item_exl2.timestamp.strftime('%H:%M:%S')}] [{log_item_exl2.level}] {log_item_exl2.message}\n" # type: ignore
                    accumulated_logs_exl2 += log_line_exl2; new_logs_update_exl2 = accumulated_logs_exl2 # type: ignore
                yield {exl2_logs_output: new_logs_update_exl2}
            except queue.Empty:
                if not exl2_thread_instance.is_alive(): break; pass
        exl2_thread_instance.join(timeout=10)

        final_status_md_parts = []
        if 'final_status_messages_list' in result_container_exl2 and result_container_exl2['final_status_messages_list']:
            final_status_md_parts.append("### EXL2 Quantization Summary:")
            for msg_item in result_container_exl2['final_status_messages_list']:
                final_status_md_parts.append(f"- {msg_item}")
            final_status_display_md = "\n".join(final_status_md_parts)
        else:
            final_status_display_md = result_container_exl2.get('final_status_message', "EXL2 process finished. Check logs.")

        err_msg_from_util = result_container_exl2.get('error_message')
        end_log_msg_exl2 = "EXL2 quantization process completed."
        end_log_lvl_exl2 = "INFO"
        if err_msg_from_util:
            end_log_msg_exl2 = f"EXL2 quantization finished with errors: {err_msg_from_util}"
            end_log_lvl_exl2 = "ERROR"
            if not (result_container_exl2.get('final_status_messages_list') and any(s.startswith("✅") for s in result_container_exl2['final_status_messages_list'])):
                final_status_display_md = f"### ❌ EXL2 Quantization Failed\nOverall Error: {err_msg_from_util}"

        if 'LogsView' in globals() and LogsView != gr.Textbox: # type: ignore
            final_log_exl2 = Log(end_log_msg_exl2, end_log_lvl_exl2, datetime.datetime.now()) # type: ignore
            accumulated_logs_exl2.append(final_log_exl2); final_log_disp_exl2 = accumulated_logs_exl2.copy() # type: ignore
        else:
            accumulated_logs_exl2 += f"[{datetime.datetime.now().strftime('%H:%M:%S')}] [{end_log_lvl_exl2}] {end_log_msg_exl2}\n"; final_log_disp_exl2 = accumulated_logs_exl2 # type: ignore
        yield {exl2_logs_output: final_log_disp_exl2, exl2_final_status_display: final_status_display_md}

    # --- Event Handlers LINKING ---
    demo.load(populate_examples_for_dropdown, outputs=[example_dropdown]) # type: ignore
    load_example_btn.click(load_selected_example_content, inputs=[example_dropdown], outputs=[merge_yaml_config])

    # Merge Tab Event Handlers
    merge_upload_to_hf.change(
        lambda upload_checked: {
            merge_repo_name: gr.update(visible=upload_checked),
            merge_private_repo: gr.update(visible=upload_checked),
            merge_hf_token_input: gr.update(visible=upload_checked)
        },
        inputs=[merge_upload_to_hf],
        outputs=[merge_repo_name, merge_private_repo, merge_hf_token_input]
    )
    merge_button.click(
        handle_merge_models,
        inputs=[
            merge_yaml_config, merge_upload_to_hf,
            merge_hf_token_input, merge_repo_name, merge_private_repo,
            merge_local_save_path,
            merge_keep_hf_cache_checkbox, merge_persistent_hf_cache_path_input,
            merge_use_gpu, merge_passthrough_option,
            gguf_model_source, exl2_model_source
        ],
        outputs=[
            merge_status_output, merge_logs_output, merged_model_path_state,
            gguf_hf_group, gguf_local_model_path_input, gguf_merged_model_display,
            exl2_hf_group, exl2_local_model_path_input, exl2_merged_model_display
        ]
    )
    merge_keep_hf_cache_checkbox.change(
        lambda x: gr.update(visible=x),
        inputs=[merge_keep_hf_cache_checkbox],
        outputs=[merge_persistent_hf_cache_path_input]
    )

    # GGUF Tab Event Handlers
    gguf_upload_to_hf.change(
        lambda upload_checked: {
            gguf_custom_name_input: gr.update(visible=upload_checked),
            gguf_private_repo: gr.update(visible=upload_checked),
            gguf_hf_token_input: gr.update(visible=upload_checked)
        },
        inputs=gguf_upload_to_hf,
        outputs=[gguf_custom_name_input, gguf_private_repo, gguf_hf_token_input]
    )
    gguf_convert_btn.click(
        handle_gguf_conversion,
        inputs=[
            gguf_model_source, merged_model_path_state, gguf_model_id, gguf_local_model_path_input,
            gguf_upload_to_hf, gguf_custom_name_input, gguf_private_repo, gguf_hf_token_input,
            gguf_local_output_path,
            gguf_q_methods, gguf_use_imatrix, gguf_imatrix_q_methods,
            gguf_use_bundled_imatrix_checkbox,
            gguf_split_model, gguf_split_max_tensors, gguf_split_max_size
        ],
        outputs=[gguf_logs_output, gguf_final_status_display] # Outputs remain the same, their visual order is changed
    )
    gguf_use_imatrix.change(lambda use_imatrix_checked: (gr.update(visible=not use_imatrix_checked), gr.update(visible=use_imatrix_checked), gr.update(visible=use_imatrix_checked)),
        inputs=gguf_use_imatrix, outputs=[gguf_q_methods, gguf_imatrix_q_methods, gguf_use_bundled_imatrix_checkbox])
    gguf_split_model.change(lambda split_checked: (gr.update(visible=split_checked), gr.update(visible=split_checked)),
        inputs=gguf_split_model, outputs=[gguf_split_max_tensors, gguf_split_max_size])

    # EXL2 Tab Event Handlers
    exl2_upload_to_hf.change(
        lambda upload_checked: {
            exl2_custom_repo_name_base: gr.update(visible=upload_checked),
            exl2_private_repo: gr.update(visible=upload_checked),
            exl2_hf_token_input: gr.update(visible=upload_checked)
        },
        inputs=exl2_upload_to_hf,
        outputs=[exl2_custom_repo_name_base, exl2_private_repo, exl2_hf_token_input]
    )
    exl2_quantize_btn.click(
        handle_exl2_quantization,
        inputs=[
            exl2_model_source, merged_model_path_state,
            exl2_hf_model_id, exl2_local_model_path_input,
            exl2_upload_to_hf, exl2_custom_repo_name_base, exl2_private_repo, exl2_hf_token_input,
            exl2_local_output_dir_base,
            exl2_bits_input_str, exl2_head_bits,
            exl2_calibration_mode, exl2_cal_dataset, exl2_cal_rows, exl2_input_measurement_file
        ],
        outputs=[exl2_logs_output, exl2_final_status_display] # Outputs remain the same, their visual order is changed
    )
    exl2_calibration_mode.change(update_exl2_calibration_visibility, inputs=[exl2_calibration_mode], outputs=[exl2_cal_dataset_group, exl2_cal_measurement_file_group]) # type: ignore

    # Model Source Visibility Handlers
    passthrough_inputs = [merged_model_path_state, gguf_model_source, exl2_model_source, merge_passthrough_option]
    passthrough_outputs = [
        gguf_hf_group, gguf_local_model_path_input, gguf_merged_model_display,
        exl2_hf_group, exl2_local_model_path_input, exl2_merged_model_display
    ]

    def update_passthrough_visibility(merged_path, gguf_choice, exl2_choice, passthrough_opt):
        use_for_gguf_exl2 = passthrough_opt == "Use for GGUF & EXL2 (Steps 2 & 3)"
        gguf_updates = update_gguf_src_visibility_wrapper(gguf_choice, merged_path if use_for_gguf_exl2 and gguf_choice == "Output from Merge Step" else None) # type: ignore
        exl2_updates = update_exl2_src_visibility_wrapper(exl2_choice, merged_path if use_for_gguf_exl2 and exl2_choice == "Output from Merge Step" else None) # type: ignore
        return {**gguf_updates, **exl2_updates}

    merged_model_path_state.change(update_passthrough_visibility, inputs=passthrough_inputs, outputs=passthrough_outputs)
    gguf_model_source.change(update_passthrough_visibility, inputs=passthrough_inputs, outputs=passthrough_outputs)
    exl2_model_source.change(update_passthrough_visibility, inputs=passthrough_inputs, outputs=passthrough_outputs)
    merge_passthrough_option.change(update_passthrough_visibility, inputs=passthrough_inputs, outputs=passthrough_outputs)


    # Initial Load
    demo.load( # type: ignore
        initial_load_updates,
        inputs=[
            merged_model_path_state, gguf_model_source, exl2_model_source,
            gguf_use_imatrix, gguf_split_model,
            merge_upload_to_hf, gguf_upload_to_hf, exl2_upload_to_hf,
            exl2_calibration_mode, merge_passthrough_option
        ],
        outputs=[
            # Merge
            merge_repo_name, merge_private_repo, merge_hf_token_input,
            # GGUF
            gguf_hf_group, gguf_local_model_path_input, gguf_merged_model_display,
            gguf_custom_name_input, gguf_private_repo, gguf_hf_token_input,
            gguf_q_methods, gguf_imatrix_q_methods, gguf_use_bundled_imatrix_checkbox,
            gguf_split_max_tensors, gguf_split_max_size,
            # EXL2
            exl2_hf_group, exl2_local_model_path_input, exl2_merged_model_display,
            exl2_custom_repo_name_base, exl2_private_repo, exl2_hf_token_input,
            exl2_cal_dataset_group, exl2_cal_measurement_file_group,
        ]
    )

if __name__ == "__main__":
    if mergekit_utils and hasattr(mergekit_utils, 'clean_tmp_folders'):
        try:
            cleaned_count = mergekit_utils.clean_tmp_folders() # type: ignore
            if cleaned_count > 0: print(f"Cleaned up {cleaned_count} old mergekit temporary item(s).")
        except Exception as e_clean: print(f"Error during initial cleanup of mergekit temp folders: {e_clean}")

    demo.queue(default_concurrency_limit=3, max_size=30).launch(debug=True, show_api=False)