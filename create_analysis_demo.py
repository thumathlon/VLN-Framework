import argparse
import json
import os
import re
import cv2
import MatterSim
import numpy as np
import math

# --- Helper Functions ---

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generates a clear, multi-page analysis demo for each decision step.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--log_file', type=str, required=True, help='Path to the detailed run.log file.')
    parser.add_argument('--pred_dir', type=str, required=True, help='Path to the directory containing case_*.json files.')
    parser.add_argument('--output_dir', type=str, default='analysis_demos', help='Directory to save the final analysis demos.')
    parser.add_argument('--root_dir', type=str, default='../datasets', help='Path to datasets root directory.')
    parser.add_argument('--fps', type=int, default=1, help='Frames per second. Use 1 to make each page last for its specified duration.')
    parser.add_argument('--create_video', action='store_true', help='Create an MP4 video for each demo.')
    parser.add_argument('--keep_frames', action='store_true', help='Keep individual frame images after creating the video.')
    parser.add_argument('--instr_id', type=str, default=None, help='(Optional) Specify a single instruction ID to process.')

    args = parser.parse_args()
    
    args.connectivity_dir = os.path.join(args.root_dir, 'R2R', 'connectivity')
    args.scan_data_dir = os.path.join(args.root_dir, 'Matterport3D', 'v1_unzip_scans')
    if not os.path.exists(args.scan_data_dir):
        raise FileNotFoundError(f"Matterport3D scan data not found at: {args.scan_data_dir}")
    
    return args

def create_simulator(args):
    sim = MatterSim.Simulator()
    sim.setDatasetPath(args.scan_data_dir)
    sim.setNavGraphPath(args.connectivity_dir)
    sim.setRenderingEnabled(True)
    sim.setCameraResolution(400, 300)
    sim.setCameraVFOV(np.deg2rad(60))
    sim.setDiscretizedViewingAngles(True)
    sim.setBatchSize(1)
    sim.initialize()
    print("INFO: MatterSim simulator initialized.")
    return sim

def create_video_from_frames(frame_paths, output_path, fps):
    if not frame_paths: return
    try:
        first_frame = cv2.imread(frame_paths[0])
        if first_frame is None: return
        height, width, _ = first_frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, float(fps), (width, height))
        for frame_path in frame_paths:
            img = cv2.imread(frame_path)
            if img is not None: video_writer.write(img)
        video_writer.release()
        print(f"SUCCESS: Video demo created: {output_path}")
    except Exception as e:
        print(f"ERROR: Failed to create video. Reason: {e}")

def parse_log_for_debug_data(log_file):
    print(f"INFO: Parsing detailed debug data from {log_file}...")
    try:
        with open(log_file, 'r') as f:
            log_content = f.read()
    except FileNotFoundError:
        print(f"FATAL: Log file '{log_file}' not found. Exiting."); exit(1)

    parsed_data = {}
    task_blocks = re.findall(r"-------------------- Environment Prompts --------------------(.*?)(?=-------------------- Environment Prompts --------------------|eval \d+ predictions)", log_content, re.DOTALL)
    
    for block in task_blocks:
        debug_match = re.search(r">>>DEBUG_STEP_JSON>>>(.*?)<<<DEBUG_STEP_JSON<<<", block)
        if debug_match:
            try:
                step_data = json.loads(debug_match.group(1))
                instr_id = step_data['instr_id']
                if instr_id not in parsed_data:
                    parsed_data[instr_id] = []
                step_data['prompt_text'] = block.split(">>>DEBUG_STEP_JSON>>>")[0].strip()
                parsed_data[instr_id].append(step_data)
            except (json.JSONDecodeError, KeyError):
                continue
                
    print(f"INFO: Parsed datastream data for {len(parsed_data)} trajectories.")
    return parsed_data

def draw_wrapped_text(image, text, start_pos, font_scale, color, line_height, width_limit):
    x, y = start_pos
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_thickness = 1
    
    words = text.split(' ')
    current_line = ""
    lines = []
    
    for word in words:
        if cv2.getTextSize(current_line + " " + word, font, font_scale, font_thickness)[0][0] < width_limit:
            current_line += " " + word
        else:
            lines.append(current_line.strip())
            current_line = word
    lines.append(current_line.strip())
    
    for line in lines:
        cv2.putText(image, line, (x, y), font, font_scale, color, font_thickness, cv2.LINE_AA)
        y += line_height
    return lines

def create_text_page(title, content, size=(1200, 900)):
    width, height = size
    page = np.zeros((height, width, 3), dtype=np.uint8)
    
    cv2.putText(page, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
    
    y_pos = 80
    prompt_lines = content.split('\n')
    for line in prompt_lines:
        draw_wrapped_text(page, line, (25, y_pos), 0.6, (255, 255, 255), 25, width - 50)
        y_pos += 25 * (1 + line.count('\n'))

    return page

def create_image_grid_panel(images_with_labels, selected_index=-1, size=(1200, 600)):
    # This function creates ONLY the top image grid part of the page.
    width, height = size
    grid_panel = np.zeros((height, width, 3), dtype=np.uint8)

    num_images = len(images_with_labels)
    if num_images == 0: return grid_panel

    grid_area_h, grid_area_w = height, width
    
    cols = 4 if num_images > 9 else 3
    if num_images <= 3: cols = num_images
    
    rows = math.ceil(num_images / cols)
    
    cell_w, cell_h = grid_area_w // cols, grid_area_h // rows
    
    img_h, img_w = int(cell_w * 0.75), cell_w
    if img_h > cell_h - 30:
        img_h, img_w = cell_h - 30, int((cell_h - 30) / 0.75)

    for i, (img, label) in enumerate(images_with_labels):
        row, col = i // cols, i % cols
        x_offset, y_offset = (cell_w - img_w) // 2, (cell_h - img_h - 30) // 2
        x_start, y_start = col * cell_w + x_offset, row * cell_h + y_offset
        
        if img is not None:
            img_resized = cv2.resize(img, (img_w, img_h))
            if i == selected_index:
                cv2.rectangle(img_resized, (0, 0), (img_w - 1, img_h - 1), (0, 255, 0), 8)
            
            grid_panel[y_start:y_start + img_h, x_start:x_start + img_w] = img_resized
            cv2.putText(grid_panel, label, (x_start + 10, y_start + img_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            
    return grid_panel

# --- Main Execution ---

def main():
    args = parse_args()
    
    pred_map = {}
    try:
        for filename in sorted(os.listdir(args.pred_dir)):
            if filename.startswith('case_') and filename.endswith('.json'):
                filepath = os.path.join(args.pred_dir, filename)
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    pred_map[data['instr_id']] = data
    except Exception as e:
        print(f"FATAL: Could not load prediction files. Error: {e}"); exit(1)
    
    datastream_map = parse_log_for_debug_data(args.log_file)
    
    ids_to_process = [args.instr_id] if args.instr_id else sorted(list(datastream_map.keys()))

    sim = create_simulator(args)
    total_processed = 0

    for instr_id in ids_to_process:
        if instr_id not in datastream_map or instr_id not in pred_map:
            continue
            
        print(f"\n---===[ Processing Prompt Demo for Trajectory: {instr_id} ]===---")
        
        pred_data = pred_map[instr_id]
        datastream_steps = datastream_map[instr_id]
        scan_id = pred_data['scan']
        
        demo_dir = os.path.join(args.output_dir, instr_id)
        os.makedirs(demo_dir, exist_ok=True)
        frame_paths = []
        
        for step_idx, step_data in enumerate(datastream_steps):
            print(f"  - Generating Pages for Step {step_idx + 1}...")
            
            # --- Page 1: Full Prompt Text ---
            prompt_title = f"Step {step_idx + 1}/{len(datastream_steps)}: Prompt Sent to LLM"
            prompt_page = create_text_page(prompt_title, step_data['prompt_text'])
            frame_path = os.path.join(demo_dir, f'step_{step_idx:02d}_page1_prompt.jpg')
            cv2.imwrite(frame_path, prompt_page)
            for _ in range(args.fps * 5): frame_paths.append(frame_path)

            # --- Page 2: Visual Inputs Grid ---
            action_options_text = re.search(r"Action options .*?:\n(.*)", step_data['prompt_text'], re.DOTALL)
            image_labels = [m.group(1) for m in re.finditer(r"^\s*([A-Z])\.", action_options_text.group(1), re.MULTILINE)] if action_options_text else []

            images_with_labels = []
            for j, cand_view in enumerate(step_data['candidate_views']):
                vp_id = cand_view['viewpointId']
                sim.newEpisode([scan_id], [vp_id], [0], [0])
                cand_img = np.array(sim.getState()[0].rgb)
                label = f"{image_labels[j]}" if j < len(image_labels) else f"Img {j+1}"
                images_with_labels.append((cand_img, f"{label}: {vp_id[:6]}..."))
            
            visual_title = f"Step {step_idx + 1}/{len(datastream_steps)}: Visual Inputs"
            # --- CRITICAL FIX: Use the correct function name ---
            visual_page = create_image_grid_panel(images_with_labels, selected_index=-1, size=(1200, 900))
            frame_path = os.path.join(demo_dir, f'step_{step_idx:02d}_page2_images.jpg')
            cv2.imwrite(frame_path, visual_page)
            for _ in range(args.fps * 5): frame_paths.append(frame_path)

            # --- Page 3: Decision & Thought Analysis Page(s) ---
            selected_vp_id = pred_data['trajectory'][step_idx + 1][0] if (step_idx + 1) < len(pred_data['trajectory']) else None
            selected_index = -1
            if selected_vp_id:
                for j, cand_view in enumerate(step_data['candidate_views']):
                    if cand_view['viewpointId'] == selected_vp_id:
                        selected_index = j; break
            
            llm_output = step_data.get('llm_full_output', "")
            thought_match = re.search(r"[#*]*\s*Thought:?\s*[#*]*(.*)", llm_output, re.DOTALL | re.IGNORECASE)
            thought = ' '.join(thought_match.group(1).strip().split()) if thought_match else "N/A"
            
            # --- CRITICAL FIX: The logic to create paginated thought pages ---
            # --- CRITICAL FIX: Use the correct function name ---
            decision_base_panel = create_image_grid_panel(images_with_labels, selected_index=selected_index, size=(1200, 600))

            font_scale, line_height, width_limit = 0.6, 25, 1150
            words = thought.split(' '); all_lines = []
            current_line = ""
            for word in words:
                if cv2.getTextSize(current_line + " " + word, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0][0] < width_limit:
                    current_line += " " + word
                else:
                    all_lines.append(current_line.strip()); current_line = word
            all_lines.append(current_line.strip())

            text_area_h = 300 - 60
            lines_per_page = text_area_h // line_height
            num_pages = math.ceil(len(all_lines) / lines_per_page) if lines_per_page > 0 else 1

            for page_num in range(num_pages):
                dashboard = np.zeros((900, 1200, 3), dtype=np.uint8)
                dashboard[0:600, :] = decision_base_panel
                
                page_title = f"Step {step_idx + 1}: Decision & Thought (Page {page_num + 1}/{num_pages})" if num_pages > 1 else f"Step {step_idx + 1}: Decision & Thought"
                cv2.putText(dashboard, page_title, (20, 620), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                
                y_pos = 650
                start_line = page_num * lines_per_page
                end_line = start_line + lines_per_page
                
                for line in all_lines[start_line:end_line]:
                    cv2.putText(dashboard, line, (25, y_pos), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
                    y_pos += line_height

                frame_path = os.path.join(demo_dir, f'step_{step_idx:02d}_page3_decision_{page_num}.jpg')
                cv2.imwrite(frame_path, dashboard)
                for _ in range(args.fps * 6): frame_paths.append(frame_path)

        if frame_paths:
            print(f"INFO: Generated {len(frame_paths)} frames for prompt demo.")
            if args.create_video:
                video_path = os.path.join(args.output_dir, f'{instr_id}_analysis_demo.mp4')
                create_video_from_frames(frame_paths, video_path, args.fps)
            if not args.keep_frames:
                for path in set(frame_paths):
                    try: os.remove(path)
                    except OSError: pass
                print(f"INFO: Cleaned up temporary frames for {instr_id}.")
            total_processed += 1

    print(f"\n---=== Finished: Processed {total_processed} trajectories. ===---")

if __name__ == '__main__':
    main()