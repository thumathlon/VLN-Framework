import sys
import numpy as np
from collections import defaultdict
from GPT.one_stage_prompt_manager import OneStagePromptManager
from .agent_base import BaseAgent
from GPT.api import gpt_infer
from utils.sam_detector import run_grounding_sam
import json
import re


class GPTNavAgent(BaseAgent):
    env_actions = {
        'left': (0, -1, 0),  # left
        'right': (0, 1, 0),  # right
        'up': (0, 0, 1),  # up
        'down': (0, 0, -1),  # down
        'forward': (1, 0, 0),  # forward
        '<end>': (0, 0, 0),  # <end>
        '<start>': (0, 0, 0),  # <start>
        '<ignore>': (0, 0, 0)  # <ignore>
    }
    for k, v in env_actions.items():
        env_actions[k] = [[vx] for vx in v]

    def __init__(self, args, env, rank=0):
        super().__init__(env)
        self.args = args

        self._build_prompt_manager()

        # Logs
        sys.stdout.flush()
        self.logs = defaultdict(list)
    
    def _build_prompt_manager(self):
        self.prompt_manager = OneStagePromptManager(self.args)
        print('Model version:', self.args.llm)

    def make_equiv_action(self, a_t, obs, traj=None):

        def take_action(i, name):
            if type(name) is int:       # Go to the next viewpoint
                self.env.env.sims[i].makeAction([name], [0], [0])
            else:                       # Adjust
                self.env.env.sims[i].makeAction(*self.env_actions[name])

        for i, ob in enumerate(obs):
            action = a_t[i]
            if action != -1:  # -1 is the <stop> action
                # Ensure the selected action index is within the valid range
                if 0 <= action < len(ob['candidate']):
                    select_candidate = ob['candidate'][action]
                    src_point = ob['viewIndex']
                    trg_point = select_candidate['pointId']
                    src_level = src_point // 12
                    trg_level = trg_point // 12
                    
                    # Vertical alignment
                    while src_level < trg_level:
                        take_action(i, 'up')
                        src_level += 1
                    while src_level > trg_level:
                        take_action(i, 'down')
                        src_level -= 1
                    
                    # Horizontal alignment
                    while self.env.env.sims[i].getState()[0].viewIndex != trg_point:
                        take_action(i, 'right')
                    
                    # Final action: move to the selected viewpoint
                    # 'idx' is the original index in 'navigableLocations', which is the correct one to use.
                    final_action_idx = select_candidate['idx']
                    take_action(i, final_action_idx)
                else:
                    print(f"Error: Action index {action} is out of range for {len(ob['candidate'])} candidates.")

                state = self.env.env.sims[i].getState()[0]
                if traj is not None:
                    traj[i]['path'].append([state.location.viewpointId])

    def rollout(self, train_ml=None, train_rl=False, reset=True):
        if reset:  # Reset env
            obs = self.env.reset()
        else:
            obs = self.env._get_obs()

        batch_size = len(obs)

        # Record the navigation path
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [[ob['viewpoint']]],
            'details': {},
            'a_t': {},
        } for ob in obs]

        if traj[0]['instr_id'] in self.results:
            return [None]

        # State variable to track the image of the current viewpoint
        current_view_image_path = None
        
        # --- Stage 1: Define Hierarchical Stopping Conditions ---
        stopping_conditions = {} # Now a dictionary
        try:
            goal_definition_prompt = (
                "You are a stop-condition analyst. Focus ONLY on the **final target scene** described at the END of the instruction; "
                "IGNORE any intermediate rooms, corridors, or maneuvers that are merely transit steps.\n"
                f"Full Instruction: \"{obs[0]['instruction']}\"\n"
                "Task: Extract two lists describing the visual requirements of that final scene and output them in a JSON object with keys "
                "\"must_have\" and \"should_have\".\n"
                "- \"must_have\": Minimal, non-negotiable visual facts that MUST be true in the final stopping viewpoint "
                "(e.g., \"at the kitchen table\", \"red chair visible\").\n"
                "- \"should_have\": Recommended, optional details that can further confirm alignment but are not strictly required "
                "(e.g., \"possibly near the target\", \"perhaps facing the target\").\n"
                "Do NOT include waypoints, hallways, or instructions about how to get there. Describe only the destination scene.\n"
                "Example – instruction: \"Walk through the hallway, go downstairs, and stop next to the black coffee machine in the kitchen\":\n"
                "{\n"
                '  \"must_have\": [\"in the kitchen\", \"black coffee machine visible\"],\n'
                '  \"should_have\": [\"very close to the coffee machine\", \"facing the coffee machine\"]\n'
                "}\n"
                "Return ONLY the JSON object."
            )
            print('--- Defining Hierarchical Stopping Conditions ---')
            goal_output, _ = gpt_infer(
                system="", text=goal_definition_prompt, image_list=[], model=self.args.llm, max_tokens=250, response_format={"type": "json_object"}
            )
            stopping_conditions = json.loads(goal_output)
            print(f'Defined Stopping Conditions: {stopping_conditions}')
        except Exception as e:
            print(f'Could not define stopping conditions: {e}')
            stopping_conditions = {} # Fallback to empty dict

        previous_angle = [{'heading': ob['heading'],
                               'elevation': ob['elevation']} for ob in obs]

        self.prompt_manager.history = ['' for _ in range(self.args.batch_size)]
        self.prompt_manager.nodes_list = [[] for _ in range(self.args.batch_size)]
        self.prompt_manager.node_imgs = [[] for _ in range(self.args.batch_size)]
        self.prompt_manager.graph = [{} for _ in range(self.args.batch_size)]
        self.prompt_manager.trajectory = [[] for _ in range(self.args.batch_size)]
        self.prompt_manager.planning = [["Navigation has just started, with no planning yet."] for _ in range(self.args.batch_size)]

        # Task 1: Strategic Planning - Extract key objects from instruction
        key_objects_list = []
        try:
            planning_prompt = (
                "You are a navigation planner. Your task is to analyze a navigation instruction. "
                "Identify all explicitly mentioned objects and scenes, as well as any other objects "
                "that are strongly associated with them (e.g., 'dining room' implies 'table' and 'chairs'). "
                "Your output must be a single, valid JSON list of strings. "
                "For example, for 'go to the dining room and find the vase', "
                "you should output: [\"dining room\", \"vase\", \"table\", \"chairs\"].\n\n"
                f"Now, analyze the following instruction: '{obs[0]['instruction']}'"
            )
            
            print('-------------------- Strategic Planning Phase --------------------')
            planning_output, _ = gpt_infer(
                system="", text=planning_prompt, image_list=[], model=self.args.llm, max_tokens=200
            )
            
            # Robustly parse JSON from the output
            cleaned_output = re.search(r'\[.*\]', planning_output, re.S)
            if cleaned_output:
                key_objects_list = json.loads(cleaned_output.group(0))
            print(f'Extracted key objects: {key_objects_list}')
        except Exception as e:
            print(f'Strategic planning failed: {e}')
            key_objects_list = []
        print('-------------------- End Strategic Planning --------------------')

        for t in range(self.args.max_action_len):
            if t == self.args.max_action_len:
                break

            # Run SAM on candidate views
            for i, ob in enumerate(obs):
                # The first image path is for the current view, if available from the last step
                if i == 0 and current_view_image_path:
                    ob['image'] = current_view_image_path
                
                for j, cand in enumerate(ob['candidate']):
                    detected_objects_in_cand = []
                    for key_object in key_objects_list:
                        detection_result = run_grounding_sam(cand['image'], key_object)
                        if detection_result is not None:
                            confidence = detection_result.get('confidence', 0.0)
                            if confidence > 0.5:
                                detected_objects_in_cand.append(f"{key_object} (conf: {confidence:.2f})")
                    cand['sam_results'] = detected_objects_in_cand
            
            cand_inputs = self.prompt_manager.make_action_prompt(obs, previous_angle)
            self.prompt_manager.key_objects_list = key_objects_list
            
            nav_input = self.prompt_manager.make_r2r_prompts(
                cand_inputs=cand_inputs, obs=obs, t=t, stopping_conditions=stopping_conditions
            )

            image_list = [cand['image'] for cand in obs[0]['candidate']]
            environment_prompts = nav_input["prompts"][0]
            
            # --- Logging Environment Prompts ---
            print('-------------------- Environment Prompts --------------------')
            print(environment_prompts)

            response_format = {"type": "json_object"} if self.args.response_format == 'json' else None
            nav_output, tokens = gpt_infer(
                nav_input["task_description"], environment_prompts, image_list,
                self.args.llm, self.args.max_tokens, response_format=response_format
            )

            # --- Logging Debug Info ---
            ob = obs[0]
            debug_step_info = {
                'instr_id': ob['instr_id'], 'step_index': t, 'current_viewpoint': ob['viewpoint'],
                'candidate_views': [{"viewpointId": cand['viewpointId'], "image_path": cand['image'], "caption": cand.get('caption', 'N/A')} for cand in ob['candidate']],
                "llm_full_output": nav_output
            }
            print(">>>DEBUG_STEP_JSON>>>" + json.dumps(debug_step_info) + "<<<DEBUG_STEP_JSON<<<")

            a_t = self.prompt_manager.parse_action(nav_output=[nav_output], only_options_batch=nav_input["only_options"], t=t)
            self.prompt_manager.parse_planning(nav_output=[nav_output])

            # --- Logging LLM Output ---
            print('-------------------- Output --------------------')
            print(nav_output)

            for i in range(batch_size):
                traj[i]['a_t'][t] = a_t[i]

            # --- Verification Step for 'stop' action ---
            is_verified_stop = False
            if a_t[0] == -1: # Agent proposes to stop
                if not current_view_image_path:
                    print("Warning: Could not find image for current viewpoint. Trusting agent's stop.")
                    is_verified_stop = True # Fail safe: if we can't see, trust the agent.
                elif stopping_conditions and isinstance(stopping_conditions, dict):
                    try:
                        print('--- Verifying Stop Condition ---')
                        must_have_objects = stopping_conditions.get('must_have', [])
                        should_have_objects = stopping_conditions.get('should_have', [])
                        
                        verification_prompt = (
                            "You are the final navigation decision-maker. An agent has proposed to STOP. You must make a final, informed decision.\n\n"
                            "**1. The Goal (defined at start):**\n"
                            f"- Must satisfy: {must_have_objects}\n"
                            f"- Should satisfy: {should_have_objects}\n\n"
                            "**2. Your Task:**\n"
                            "a. Look at the image. Does the scene satisfy the 'must_have' conditions?\n"
                            "b. How well does it satisfy the 'should_have' conditions?\n"
                            "c. Based on this, make a final decision.\n\n"
                            "Your answer MUST be a JSON object with TWO fields:\n"
                            "{\n"
                            '    "verification_passed": boolean, \n'
                            '    "reasoning": "Explain your final decision."\n'
                            "}"
                        )
                        
                        verification_output, _ = gpt_infer(
                            system="", text=verification_prompt, image_list=[current_view_image_path], 
                            model=self.args.llm, max_tokens=400, response_format={"type": "json_object"}
                        )
                        
                        verification_result = json.loads(verification_output)
                        print(f"Verification Result: {verification_result}")

                        if verification_result.get("verification_passed", False):
                            is_verified_stop = True
                            print("Stop condition VERIFIED.")
                        else:
                            reason = verification_result.get("reasoning", "Verification failed.")
                            self.prompt_manager.history[0] += f" (Step {t} Stop Rejected: {reason})"
                            print(f"Stop condition REJECTED: {reason}")
                    except Exception as e:
                        print(f"Stop verification failed with an error: {e}")
                else:
                    is_verified_stop = True # No conditions, or conditions are not a dict, accept stop
            
            if is_verified_stop:
                print("Final decision: STOP")
                break
            
            # State Management: Update current_view_image_path for the NEXT step's verification.
            if a_t[0] != -1 and 0 <= a_t[0] < len(obs[0]['candidate']):
                 current_view_image_path = obs[0]['candidate'][a_t[0]]['image']

            # Execute action and get new observation
            self.make_equiv_action(a_t, obs, traj)
            obs = self.env._get_obs()

            previous_angle = [{'heading': ob['heading'], 'elevation': ob['elevation']} for ob in obs]
            self.prompt_manager.make_history(a_t, nav_input, t)

        return traj
