import re
import math


class OneStagePromptManager(object):
    def __init__(self, args):

        self.args = args
        self.history  = ['' for _ in range(self.args.batch_size)]
        self.nodes_list = [[] for _ in range(self.args.batch_size)]
        self.node_imgs = [[] for _ in range(self.args.batch_size)]
        self.graph  = [{} for _ in range(self.args.batch_size)]
        self.trajectory = [[] for _ in range(self.args.batch_size)]
        self.planning = [["Navigation has just started, with no planning yet."] for _ in range(self.args.batch_size)]
        self.key_objects_list = []  # Strategic planning key objects list

    def get_action_concept(self, rel_heading, rel_elevation):
        # Initialize to satisfy static analyzers (ensure always defined)
        action_text: str = ""
        if rel_elevation > 0:
            action_text = 'go up'
        elif rel_elevation < 0:
            action_text = 'go down'
        else:
            if rel_heading < 0:
                if rel_heading >= -math.pi / 2:
                    action_text = 'turn left'
                elif rel_heading < -math.pi / 2 and rel_heading > -math.pi * 3 / 2:
                    action_text = 'turn around'
                else:
                    action_text = 'turn right'
            elif rel_heading > 0:
                if rel_heading <= math.pi / 2:
                    action_text = 'turn right'
                elif rel_heading > math.pi / 2 and rel_heading < math.pi * 3 / 2:
                    action_text = 'turn around'
                else:
                    action_text = 'turn left'
            elif rel_heading == 0:
                action_text = 'go forward'

        return action_text

    def make_action_prompt(self, obs, previous_angle):

        nodes_list, graph, trajectory, node_imgs = self.nodes_list, self.graph, self.trajectory, self.node_imgs

        batch_view_lens, batch_cand_vpids = [], []
        batch_cand_index = []
        batch_action_prompts = []

        for i, ob in enumerate(obs):
            cand_vpids = []
            cand_index = []
            action_prompts = []

            if ob['viewpoint'] not in nodes_list[i]:
                # update nodes list (place 0)
                nodes_list[i].append(ob['viewpoint'])
                node_imgs[i].append(None)

            # update trajectory
            trajectory[i].append(ob['viewpoint'])

            # cand views
            for j, cc in enumerate(ob['candidate']):

                cand_vpids.append(cc['viewpointId'])
                cand_index.append(cc['pointId'])
                direction = self.get_action_concept(cc['absolute_heading'] - previous_angle[i]['heading'],
                                                          cc['absolute_elevation'] - 0)

                if cc['viewpointId'] not in nodes_list[i]:
                    nodes_list[i].append(cc['viewpointId'])
                    node_imgs[i].append(cc['image'])
                    node_index = nodes_list[i].index(cc['viewpointId'])
                else:
                    node_index = nodes_list[i].index(cc['viewpointId'])
                    node_imgs[i][node_index] = cc['image']

                # Task 3: Concise SAM summary in action prompts
                sam_results = cc.get('sam_results', [])
                if sam_results:
                    # 结果格式如 "stairs (0.70)"
                    sam_text = ', '.join(sam_results)
                    action_text = f"{direction} to Place {node_index} (Image {j}: {sam_text})"
                else:
                    # 无关键物体时省略冗余说明，避免 prompt 过长
                    action_text = f"{direction} to Place {node_index} (Image {j})"
                action_prompts.append(action_text)

            batch_cand_index.append(cand_index)
            batch_cand_vpids.append(cand_vpids)
            batch_action_prompts.append(action_prompts)

            # update graph
            if ob['viewpoint'] not in graph[i].keys():
                graph[i][ob['viewpoint']] = cand_vpids

        return {
            'cand_vpids': batch_cand_vpids,
            'cand_index':batch_cand_index,
            'action_prompts': batch_action_prompts,
        }

    def make_action_options(self, cand_inputs, t):
        action_options_batch = []  # complete action options
        only_options_batch = []  # only option labels
        batch_action_prompts = cand_inputs["action_prompts"]
        batch_size = len(batch_action_prompts)

        for i in range(batch_size):
            action_prompts = batch_action_prompts[i]
            if bool(self.args.stop_after):
                if t >= self.args.stop_after:
                    stop_action_text = 'stop (Use ONLY if perfectly aligned with the final instruction)'
                    action_prompts = [stop_action_text] + action_prompts

            full_action_options = [chr(j + 65)+'. '+action_prompts[j] for j in range(len(action_prompts))]
            only_options = [chr(j + 65) for j in range(len(action_prompts))]
            action_options_batch.append(full_action_options)
            only_options_batch.append(only_options)

        return action_options_batch, only_options_batch

    def make_history(self, a_t, nav_input, t):
        batch_size = len(a_t)
        for i in range(batch_size):
            # 直接使用原始 only_actions 列表，避免人为插入 'stop' 导致索引错位
            # Robust indexing: fall back to first option if index is out of range
            if a_t[i] < len(nav_input["only_actions"][i]):
                last_action = nav_input["only_actions"][i][a_t[i]]
            else:
                last_action = nav_input["only_actions"][i][0]  # avoid IndexError
            if t == 0:
                self.history[i] += f"""step {str(t)}: {last_action}"""
            else:
                self.history[i] += f""", step {str(t)}: {last_action}"""

    def make_map_prompt(self, i):
        # graph-related text
        trajectory = self.trajectory[i]
        nodes_list = self.nodes_list[i]
        graph = self.graph[i]

        no_dup_nodes = []
        trajectory_text = 'Place'
        graph_text = ''

        candidate_nodes = graph[trajectory[-1]]

        # trajectory and map connectivity
        for node in trajectory:
            node_index = nodes_list.index(node)
            trajectory_text += f""" {node_index}"""

            if node not in no_dup_nodes:
                no_dup_nodes.append(node)

                adj_text = ''
                adjacent_nodes = graph[node]
                for adj_node in adjacent_nodes:
                    adj_index = nodes_list.index(adj_node)
                    adj_text += f""" {adj_index},"""

                graph_text += f"""\nPlace {node_index} is connected with Places{adj_text}"""[:-1]

        # ghost nodes info
        graph_supp_text = ''
        supp_exist = None
        for node_index, node in enumerate(nodes_list):

            if node in trajectory or node in candidate_nodes:
                continue
            supp_exist = True
            graph_supp_text += f"""\nPlace {node_index}, which is corresponding to Image {node_index}"""

        if supp_exist is None:
            graph_supp_text = """Nothing yet."""

        return trajectory_text, graph_text, graph_supp_text

    def make_r2r_prompts(self, obs, cand_inputs, t, stopping_conditions=None):

        # Task 3: Restructured prompt for forced comprehensive observation
        background = """You are a navigation agent in a 3D environment. Your goal is to follow the instruction precisely."""
        
        init_history = 'The navigation has just begun, with no history.'
        batch_size = len(obs)
        action_options_batch, only_options_batch = self.make_action_options(cand_inputs, t=t)
        prompt_batch = []
        only_actions_batch = []
        
        for i in range(batch_size):
            instruction = obs[i]["instruction"]
            trajectory_text, graph_text, graph_supp_text = self.make_map_prompt(i)
            
            # Get key objects list for strategic plan
            key_objects_list = getattr(self, 'key_objects_list', [])
            key_objects_str = ', '.join(key_objects_list) if key_objects_list else 'No key objects identified'
            
            # Format stopping conditions for display
            if stopping_conditions and isinstance(stopping_conditions, dict):
                must_have_str = ', '.join(stopping_conditions.get('must_have', [])) or "None"
                should_have_str = ', '.join(stopping_conditions.get('should_have', [])) or "None"
                stopping_cond_str = f"Must Have: [{must_have_str}], Should Have: [{should_have_str}]"
            else:
                stopping_cond_str = 'Not yet defined.'

            # Create two-stage prompt structure
            task_description = f"""You are a navigation agent in a 3D environment. Your goal is to follow the instruction precisely.

**Instruction**: {instruction}
**Stopping Conditions**: {stopping_cond_str}
**Key Objects (fixed)**: {key_objects_str}

**History**: {self.history[i] if t > 0 else init_history}
**Trajectory**: {trajectory_text}
**Map**: {graph_text}
**Supplementary Info**: {graph_supp_text}
**Previous Planning**: {self.planning[i][-1]}

**Guiding Principles for Superior Navigation**:
- **Goal Completion is Key**: Your primary objective is to fulfill the *entire* instruction, not just to reach a visually similar area.
- **The Final Step Matters**: Before choosing 'stop', always perform a final check: could a small adjustment (like turning or one more step) better align you with the instruction's most specific detail? Stopping too early is a failure.

**Your Task (Step {t})**:
To make the best decision, you must complete two steps:
1. **Step 1: Describe ALL Options.** First, describe the image for EACH action option below. **Note: If an option is 'stop', it has no associated image; simply state this fact in your description.** Do not reference vision system results in this step.
2. **Step 2: Decide.** Now, combine the vision system results with your image descriptions, provide your comparative thought process, and choose the best action.

**--- Step 1: Describe ALL Options ---**

**Action options**:"""
            
            # Add each action option with vision system results
            for j, action_option in enumerate(action_options_batch[i]):
                # action_option 已带字母前缀，如 "A. turn left …"，无需再重复添加
                task_description += f"\n{action_option}"
            
            task_description += f"""

**Your Descriptions**:"""
            
            # Add placeholders for LLM to fill in descriptions
            for j, action_option in enumerate(action_options_batch[i]):
                letter = chr(j + 65)
                task_description += f"\n- **Description for Option {letter}**: (Your description for the image of Option {letter}, or a note if it is the 'stop' action)"
            
            task_description += f"""

**--- Step 2: Decide ---**

**Thought**: (Based on your descriptions above and the vision system results, provide your final comparative thought process here)
**New Planning**: (Update your multi-step path planning based on current observations)
**Action**: (Choose one letter: {', '.join(only_options_batch[i])})"""

            # Strip the leading “X. ” to form raw action strings list (keeps any prepended “stop”)
            raw_actions = [opt.split('. ', 1)[1] for opt in action_options_batch[i]]
            only_actions_batch.append(raw_actions)

            prompt_batch.append(task_description)  # Send full prompt to LLM

        nav_input = {
            "task_description": background,  # keep concise system prompt
            "prompts": prompt_batch,
            "only_options": only_options_batch,
            "action_options": action_options_batch,
            # Use raw_actions aligned with any prepended 'stop', preventing index mismatch
            "only_actions": only_actions_batch
        }

        return nav_input

    def make_r2r_json_prompts(self, obs, cand_inputs, t):

        background = """You are an embodied robot that navigates in the real world."""
        background_supp = """You need to explore between some places marked with IDs and ultimately find the destination to stop.""" \
        + """ At each step, a series of images corresponding to the places you have explored and have observed will be provided to you."""

        instr_des = """'Instruction' is a global, step-by-step detailed guidance, but you might have already executed some of the commands. You need to carefully discern the commands that have not been executed yet."""
        traj_info = """'Trajectory' represents the ID info of the places you have explored. You start navigating from Place 0."""
        map_info = """'Map' refers to the connectivity between the places you have explored and other places you have observed."""
        map_supp = """'Supplementary Info' records some places and their corresponding images you have ever seen but have not yet visited. These places are only considered when there is a navigation error, and you decide to backtrack for further exploration."""
        history = """'History' represents the places you have explored in previous steps along with their corresponding images. It may include the correct landmarks mentioned in the 'Instruction' as well as some past erroneous explorations."""
        option = """'Action options' are some actions that you can take at this step."""
        pre_planning = """'Previous Planning' records previous long-term multi-step planning info that you can refer to now."""

        requirement = """For each provided image of the places, you should combine the 'Instruction' and carefully examine the relevant information, such as scene descriptions, landmarks, and objects. You need to align 'Instruction' with 'History' (including corresponding images) to estimate your instruction execution progress and refer to 'Map' for path planning. Check the Place IDs in the 'History' and 'Trajectory', avoiding repeated exploration that leads to getting stuck in a loop, unless it is necessary to backtrack to a specific place."""
        dist_require = """If you can already see the destination, estimate the distance between you and it. If the distance is far, continue moving and try to stop within 1 meter of the destination."""
        thought = """Your answer should be JSON format and must include three fields: 'Thought', 'New Planning', and 'Action'. You need to combine 'Instruction', 'Trajectory', 'Map', 'Supplementary Info', your past 'History', 'Previous Planning', 'Action options', and the provided images to think about what to do next and why, and complete your thinking into 'Thought'."""
        new_planning = """Based on your 'Map', 'Previous Planning' and current 'Thought', you also need to update your new multi-step path planning to 'New Planning'."""
        action = """At the end of your output, you must provide a single capital letter in the 'Action options' that corresponds to the action you have decided to take, and place only the letter into 'Action', such as "Action: A"."""

        task_description = f"""{background} {background_supp}\n{instr_des}\n{history}\n{traj_info}\n{map_info}\n{map_supp}\n{pre_planning}\n{option}\n{requirement}\n{dist_require}\n{thought}\n{new_planning}\n{action}"""

        init_history = 'The navigation has just begun, with no history.'

        batch_size = len(obs)
        action_options_batch, only_options_batch = self.make_action_options(cand_inputs, t=t)
        prompt_batch = []
        for i in range(batch_size):
            instruction = obs[i]["instruction"]

            trajectory_text, graph_text, graph_supp_text = self.make_map_prompt(i)

            if t == 0:
                prompt = f"""Instruction: {instruction}\nHistory: {init_history}\nTrajectory: {trajectory_text}\nMap:{graph_text}\nSupplementary Info: {graph_supp_text}\nPrevious Planning:\n{self.planning[i][-1]}\nAction options (step {str(t)}): {action_options_batch[i]}"""
            else:
                prompt = f"""Instruction: {instruction}\nHistory: {self.history[i]}\nTrajectory: {trajectory_text}\nMap:{graph_text}\nSupplementary Info: {graph_supp_text}\nPrevious Planning:\n{self.planning[i][-1]}\nAction options (step {str(t)}): {action_options_batch[i]}"""

            prompt_batch.append(prompt)

        nav_input = {
            "task_description": task_description,
            "prompts" : prompt_batch,
            "only_options": only_options_batch,
            "action_options": action_options_batch,
            "only_actions": cand_inputs["action_prompts"]
        }

        return nav_input

    def parse_planning(self, nav_output):
        """
        Only supports parsing outputs in the style of GPT-4v.
        Please modify the parsers if the output style is inconsistent.
        """
        batch_size = len(nav_output)
        keyword1 = '\nNew Planning:'
        keyword2 = '\nAction:'
        for i in range(batch_size):
            output = nav_output[i].strip()
            start_index = output.find(keyword1) + len(keyword1)
            end_index = output.find(keyword2)

            if output.find(keyword1) < 0 or start_index < 0 or end_index < 0 or start_index >= end_index:
                planning = "No plans currently."
            else:
                planning = output[start_index:end_index].strip()

            planning = planning.replace('new', 'previous').replace('New', 'Previous')

            self.planning[i].append(planning)

        return planning

    def parse_json_planning(self, json_output):
        try:
            planning = json_output["New Planning"]
        except:
            planning = "No plans currently."

        self.planning[0].append(planning)
        return planning

    def parse_action(self, nav_output, only_options_batch, t):
        """
        Only supports parsing outputs in the style of GPT-4v.
        Please modify the parsers if the output style is inconsistent.
        """
        batch_size = len(nav_output)
        output_batch = []
        output_index_batch = []

        for i in range(batch_size):
            output = nav_output[i].strip()

            # Allow variations like "**Action**:" or "### Action:" by stripping markdown
            pattern = re.compile(r"Action", re.IGNORECASE)  # keyword
            matches = pattern.finditer(output)
            indices = [match.start() for match in matches]
            if indices:
                output_section = output[indices[-1]:]
            else:
                # Keyword “Action” not found – fall back to full text to avoid IndexError
                output_section = output

            # Accept formats like "Action: A", "**Action**: B", "Action - C"
            search_result = re.findall(
                r"Action[^A-Za-z0-9]{0,5}:\s*([A-M])",
                output_section,
                re.I
            )
            if search_result:
                output = search_result[-1]

                if output in only_options_batch[i]:
                    output_batch.append(output)
                    output_index = only_options_batch[i].index(output)
                    output_index_batch.append(output_index)
                else:
                    output_index = 0
                    output_index_batch.append(output_index)
            else:
                output_index = 0
                output_index_batch.append(output_index)

        # 若设置 stop_after 并且当前步数已达到阈值，则 action_options 首位为 “stop”，
        # 解析出的索引需要与 candidate 视角对齐：
        #   0 → 停止 (-1 表示 stop)；其余索引减 1 才对应候选 id
        if bool(self.args.stop_after) and t >= self.args.stop_after:
            adjusted_index_batch = []
            for idx in output_index_batch:
                if idx == 0:
                    adjusted_index_batch.append(-1)  # 专用 -1 表示 stop
                else:
                    adjusted_index_batch.append(idx - 1)
            output_index_batch = adjusted_index_batch

        return output_index_batch

    def parse_json_action(self, json_output, only_options_batch, t):
        try:
            output = str(json_output["Action"])
            if output in only_options_batch[0]:
                output_index = only_options_batch[0].index(output)
            else:
                output_index = 0

        except:
            output_index = 0

        # No additional offset; stop option already at correct index

        output_index_batch = [output_index]
        return output_index_batch
