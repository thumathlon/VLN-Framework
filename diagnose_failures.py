import argparse
import json
import os
import re
from collections import Counter

# --- Helper Functions ---

def parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnoses failure modes of navigation trajectories based on log and prediction files.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--log_file', type=str, required=True, help='Path to the detailed run.log file.')
    parser.add_argument('--pred_dir', type=str, required=True, help='Path to the directory containing case_*.json files.')
    parser.add_argument('--output_file', type=str, default='failure_diagnosis_report.txt', help='File to save the diagnosis report.')

    args = parser.parse_args()
    return args

def load_all_data(log_file, pred_dir):
    """Loads and merges all necessary data from log and prediction files."""
    print("--- Loading and Merging Data ---")
    
    # 1. Load prediction data (trajectory, evaluation metrics, etc.)
    pred_map = {}
    try:
        for filename in sorted(os.listdir(pred_dir)):
            if filename.startswith('case_') and filename.endswith('.json'):
                filepath = os.path.join(pred_dir, filename)
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    pred_map[data['instr_id']] = data
        print(f"INFO: Loaded {len(pred_map)} prediction files.")
    except Exception as e:
        print(f"FATAL: Could not load prediction files from '{pred_dir}'. Error: {e}"); exit(1)

    # 2. Load debug data from the log file
    print(f"INFO: Parsing detailed debug data from {log_file}...")
    try:
        with open(log_file, 'r') as f:
            log_content = f.read()
    except FileNotFoundError:
        print(f"FATAL: Log file '{log_file}' not found. Exiting."); exit(1)

    debug_data_map = {}
    task_blocks = re.split(r"-------------------- Environment Prompts --------------------", log_content)

    for block in task_blocks:
        if not block.strip(): continue
        debug_matches = re.findall(r">>>DEBUG_STEP_JSON>>>(.*?)<<<DEBUG_STEP_JSON<<<", block)
        if debug_matches:
            try:
                first_step_data = json.loads(debug_matches[0])
                instr_id = first_step_data['instr_id']
                if instr_id not in debug_data_map:
                    debug_data_map[instr_id] = []
                for match in debug_matches:
                    debug_data_map[instr_id].append(json.loads(match))
            except (json.JSONDecodeError, KeyError):
                continue
    print(f"INFO: Parsed debug data for {len(debug_data_map)} trajectories.")
    
    return pred_map, debug_data_map


def diagnose_failures(pred_map, debug_data_map):
    """Analyzes each failed trajectory and categorizes the failure mode."""
    print("\n--- Diagnosing Failure Modes ---")
    
    failure_summary = {
        "Early Stopping": [],
        "Initial Misunderstanding": [],
        "Looping": [],
        "Severe Off-track": [],
        "Other/Complex": []
    }
    
    for instr_id, pred_data in pred_map.items():
        is_success = pred_data.get('evaluation', {}).get('success', [0.0])[0] == 1.0
        
        # We only diagnose failed cases
        if is_success:
            continue
            
        # Get all relevant data for this failed case
        oracle_success = pred_data.get('evaluation', {}).get('oracle_success', [0.0])[0] == 1.0
        nav_error = pred_data.get('evaluation', {}).get('nav_error', [99.0])[0]
        trajectory = [p[0] for p in pred_data.get('trajectory', [])]
        
        # --- Diagnosis Rules ---
        
        # Rule 1: Early Stopping
        # Condition: The agent *could have* succeeded (Oracle Success is true), but it failed.
        # This is the most direct measure of stopping at the wrong time.
        if oracle_success:
            failure_summary["Early Stopping"].append(instr_id)
            continue # Prioritize this diagnosis

        # Rule 2: Looping
        # Condition: The number of unique viewpoints is significantly less than the number of steps.
        # We define "significant" as visiting less than 70% unique viewpoints.
        if len(trajectory) > 3: # Looping requires at least a few steps
            unique_viewpoints = len(set(trajectory))
            revisit_ratio = (len(trajectory) - unique_viewpoints) / len(trajectory)
            if revisit_ratio > 0.3: # More than 30% of steps are revisits
                failure_summary["Looping"].append(instr_id)
                continue

        # Rule 3: Initial Misunderstanding
        # Condition: The agent's first move was away from the first move of the ground truth path.
        # We check if the agent's second location is on the ground truth path at all.
        gt_path = pred_data.get('gt_traj', [])
        if len(trajectory) > 1 and len(gt_path) > 1:
            agent_first_move = trajectory[1]
            if agent_first_move not in gt_path:
                failure_summary["Initial Misunderstanding"].append(instr_id)
                continue

        # Rule 4: Severe Off-track
        # Condition: If none of the above, but the final nav_error is very large.
        # This catches cases where the agent just gets completely lost.
        if nav_error > 8.0: # Threshold for being severely lost
            failure_summary["Severe Off-track"].append(instr_id)
            continue
            
        # If none of the specific rules match, classify as 'Other'
        failure_summary["Other/Complex"].append(instr_id)
        
    return failure_summary

def generate_report(failure_summary, output_file):
    """Generates a text report of the failure diagnosis."""
    print(f"\n--- Generating Report to {output_file} ---")
    
    total_failures = sum(len(v) for v in failure_summary.values())
    
    with open(output_file, 'w') as f:
        f.write("="*50 + "\n")
        f.write("Navigation Failure Diagnosis Report\n")
        f.write("="*50 + "\n\n")
        
        f.write(f"Total Failed Trajectories Analyzed: {total_failures}\n\n")
        
        f.write("--- Failure Categories ---\n\n")
        
        for category, ids in failure_summary.items():
            count = len(ids)
            percentage = (count / total_failures) * 100 if total_failures > 0 else 0
            f.write(f"Category: {category}\n")
            f.write(f"  Count: {count}\n")
            f.write(f"  Percentage: {percentage:.1f}%\n")
            if ids:
                f.write(f"  Instruction IDs: {', '.join(ids)}\n")
            f.write("-" * 30 + "\n\n")
            
    print("Report generated successfully.")


# --- Main Execution ---

if __name__ == '__main__':
    args = parse_args()
    
    # 1. Load all data without modifying any source code
    pred_map, debug_data_map = load_all_data(args.log_file, args.pred_dir)
    
    # 2. Diagnose failures based on the loaded data
    failure_summary = diagnose_failures(pred_map, debug_data_map)
    
    # 3. Generate a human-readable report
    generate_report(failure_summary, args.output_file)