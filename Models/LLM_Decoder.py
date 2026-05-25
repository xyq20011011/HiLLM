from Models.LLM_utils import LLM_agent
import torch
import json
import re
from tqdm import tqdm


class LLM_Decoder:
    def __init__(self, q_matrix, p_text_path, final_tree=None):
        self.agent = LLM_agent(system_prompt=(
            "你是一名学生作答模拟代理。\n"
            "系统会给出你的能力水平（proficiency）以及若干道题目。(能力水平仅代表学生在对应知识上的得分，不代表答对相关试题的概率), "
            "能力水平会展现学生在不同层级的知识点的掌握情况，使用嵌套的方式描述学生在知识点树上的得分\n"
            "你的任务是根据自身能力水平，模拟在这些题目上的作答表现。\n\n"
            "【说明】\n"
            "1. 你需要理解每道题目的内容和难度。\n"
            "2. 根据题目难度与自己的能力水平，决定是否能答对。\n"
            "3. 你的输出应为一个 JSON 格式的列表，其中包含每道题目的作答结果。\n"
            "4. 每个结果应包含：题目ID、答对概率（0 到 1之间）、简要作答理由（例如：'概念理解不全'、'计算失误'、'熟练掌握' 等）。\n\n"
            "【示例输出】\n"
            "[\n"
            "  {\"qid\": 1, \"response\": 0.76, \"reason\": \"掌握了相关知识点\"},\n"
            "  {\"qid\": 2, \"response\": 0。31, \"reason\": \"对该概念理解不充分\"}\n"
            "]\n\n"
            "请严格按照上述格式输出。不要解释，不要添加额外文本，必须保证输出的文本可以被json解析，不要用错{}和[]。"
        ))
        self.Q_matrix = q_matrix.cuda()
        with open(p_text_path, "r", encoding="utf-8") as f:
            self.problem_text = json.load(f)
        self.final_tree = final_tree
        self.concept_text = {}

        stack = [final_tree]
        while stack:
            node = stack.pop()
            node_id = node["id"]
            node_name = node.get("name", "")

            self.concept_text[node_id] = node_name

            children = node.get("children", [])
            stack.extend(children)

    def __call__(self, p_matrix, theta, mask):
        batch_size = mask.shape[0]
        theta = torch.sigmoid(theta)
        LLM_predicted_p_matrix = torch.zeros_like(p_matrix, dtype=torch.float32)
        for stu_idx in range(batch_size):
            predict_pid = torch.where(mask[stu_idx, :])[0]
            valid_theta_idx = torch.where((p_matrix[stu_idx] @ self.Q_matrix) != 0)[0]

            self.node_dict = {}

            def flatten_tree(node):
                self.node_dict[node["id"]] = node
                for child in node.get("children", []):
                    flatten_tree(child)

            flatten_tree(self.final_tree)

            def build_proficiency_tree(node):
                lines = []
                children_lines = []
                for child in node.get("children", []):
                    child_str = build_proficiency_tree(child)
                    if child_str:  # 仅保留有效子树
                        children_lines.append(child_str)

                is_valid_leaf = node["id"] in [idx.item() for idx in valid_theta_idx]

                if not children_lines and not is_valid_leaf:
                    return ""  # 子孙都不在 valid_theta_idx，跳过

                if is_valid_leaf:
                    leaf_score = theta[stu_idx, torch.tensor([node["id"]], device=theta.device)].item()
                    leaf_str = f"{node['name']}: {leaf_score:.3f}"
                else:
                    leaf_str = node['name']

                if children_lines:
                    inner = ", ".join(children_lines)
                    return f"{leaf_str} {{{inner}}}"
                else:
                    return leaf_str

            proficiency = build_proficiency_tree(self.final_tree)

            p_ids = []
            problem_text_list = []
            for problem_idx in predict_pid:
                p_ids.append(problem_idx.item())
                problem_text_list.append(self.problem_text[str(problem_idx.item())])

            problems_str = "\n".join([
                f"{i+1}. (QID={qid}) {text}"
                for i, (qid, text) in enumerate(zip(p_ids, problem_text_list))
            ])

            prompt = (
                f"学生的能力水平描述如下：\n{proficiency}\n\n"
                f"以下是需要作答的题目：\n{problems_str}\n\n"
                "请你根据学生能力水平模拟在这些题目上的作答情况，"
                "并以 JSON 格式输出结果。"
            )

            raw_output = self.agent.ask(prompt)


            try:
                responses = json.loads(raw_output)
            except json.JSONDecodeError:
                try:
                    raw_output = raw_output[raw_output.find("["):raw_output.rfind("]")+1]
                    responses = json.loads(raw_output)
                except Exception:
                    print("⚠️ LLM 输出解析失败，原始输出：", raw_output)
                    responses = []

            for predicted_response_dict in responses:
                qid = predicted_response_dict["qid"]
                response = predicted_response_dict["response"]
                LLM_predicted_p_matrix[stu_idx, qid] = response

        assert torch.max(LLM_predicted_p_matrix) <= 1 and torch.min(LLM_predicted_p_matrix) >= 0, (torch.max(LLM_predicted_p_matrix), torch.min(LLM_predicted_p_matrix))
        return LLM_predicted_p_matrix


class LLM_Encoder:
    def __init__(self, q_matrix, p_text_path, final_tree=None):
        self.agent = LLM_agent(system_prompt=(
            "你是一名学生学习诊断分析代理。\n"
            "系统会给出学生在若干题目上的作答记录，以及题目所对应的知识点信息和知识点的层级结构。\n"
            "你的任务是基于学生的作答表现，推断其在各知识点上的能力水平（proficiency）。\n\n"

            "【能力水平说明】\n"
            "1. proficiency 表示学生对对应知识点的掌握程度，取值范围为 0 到 1。\n"
            "2. proficiency 不是答对概率，而是对知识点理解水平的综合反映。\n"
            "3. 能力水平需要按照给定的知识点树结构进行描述，使用嵌套方式表示不同层级知识点的掌握情况。\n\n"

            "【任务要求】\n"
            "1. 你需要理解每道题目的内容、所涉及的知识点以及学生的作答结果。\n"
            "2. 根据学生在相关题目上的作答表现，推断其对应知识点的掌握程度。\n"
            "3. 仅对题目中出现过的知识点进行评估，不要臆测未出现的知识点。\n"
            "4. 将知识点树中的每一个 '?' 替换为具体的数值（0-1），不保留 '?'。\n"
            "5. 输出仍需保持知识点树的嵌套结构。\n\n"

            "【输出格式】\n"
            "1. 输出的文本应严格按照原树结构，只替换 '?' 为数值。\n"
            "2. 输出示例：\n"
            "计算机科学与技术: 0.78 {\n"
            "    存储与内存管理: 0.65 {\n"
            "        文件系统结构: 0.70 {\n"
            "            数据存储结构: 0.66 {记录结构: 0.68}\n"
            "        },\n"
            "        内存管理: 0.72 {内存管理: 0.71 {分页: 0.69, 段式存储管理: 0.67}}\n"
            "    },\n"
            "    算法与数学基础: 0.80 { ... }\n"
            "}\n\n"

            "请严格按照原知识点树格式输出，只替换 '?' 为数值，"
            "不要添加任何额外解释或文本，必须保证输出可以解析为嵌套的数值树。"
        ))
        self.Q_matrix = q_matrix.cuda()
        with open(p_text_path, "r", encoding="utf-8") as f:
            self.problem_text = json.load(f)
        self.final_tree = final_tree
        self.concept_text = {}

        stack = [final_tree]
        while stack:
            node = stack.pop()
            node_id = node["id"]
            node_name = node.get("name", "")

            self.concept_text[node_id] = node_name

            children = node.get("children", [])
            stack.extend(children)

    def __call__(self, p_matrix, theta, mask):
        batch_size = mask.shape[0]
        LLM_theta = torch.ones_like(theta) * 0.5
        for stu_idx in tqdm(range(batch_size)):
            predict_pid = torch.where(mask[stu_idx, :])[0]
            valid_theta_idx = torch.where((p_matrix[stu_idx] @ self.Q_matrix) != 0)[0]

            self.node_dict = {}

            def flatten_tree(node):
                self.node_dict[node["id"]] = node
                for child in node.get("children", []):
                    flatten_tree(child)

            flatten_tree(self.final_tree)

            theta_id_map = {}

            def build_proficiency_tree(node):
                lines = []
                # 递归子节点
                children_lines = []
                for child in node.get("children", []):
                    child_str = build_proficiency_tree(child)
                    if child_str:  # 仅保留有效子树
                        children_lines.append(child_str)

                # 判断自己是否是 valid leaf
                is_valid_leaf = node["id"] in [idx.item() for idx in valid_theta_idx]

                if not children_lines and not is_valid_leaf:
                    return ""  # 子孙都不在 valid_theta_idx，跳过

                # 叶子结点显示分数
                if is_valid_leaf:
                    leaf_str = f"{node['name']}: ?"
                    theta_id_map[node['name']] = torch.tensor([node["id"]], device=theta.device)
                else:
                    leaf_str = node['name']

                if children_lines:
                    # 有子节点，嵌套大括号
                    inner = ", ".join(children_lines)
                    return f"{leaf_str} {{{inner}}}"
                else:
                    return leaf_str

            # 生成文本
            proficiency = build_proficiency_tree(self.final_tree)

            p_ids = []
            problems_str = ""
            for problem_idx in predict_pid:
                p_ids.append(problem_idx.item())
                correct_idx = p_matrix[stu_idx, problem_idx.item()]
                if correct_idx.item() == 1:
                    problems_str += ("\n回答错误：" + self.problem_text.get(str(problem_idx.item()), ""))
                elif correct_idx.item() == 2:
                    problems_str += ("\n回答正确：" + self.problem_text.get(str(problem_idx.item()), ""))
                else:
                    raise ValueError

            # 构造提示
            prompt = (
                f"学生需要评估的能力树如下：\n{proficiency}\n\n"
                f"以下是学生的做题表现：\n{problems_str}\n\n"
                "请你根据学生的做题表现诊断学生的能力，填补能力树中?的部分，返回同样的能力树，仅仅将？替换为你诊断出的0-1数值"
            )

            def extract_concept_scores(prof_str):

                # 匹配形式: 概念名: 数字
                pattern = re.compile(r'([\w\s\d\-/&]+):\s*([\d.]+)')
                results = []
                for match in pattern.finditer(prof_str):
                    concept, score = match.groups()
                    results.append((concept.strip(), float(score)))
                return results


            raw_output = self.agent.ask(prompt)
            concept_scores = extract_concept_scores(raw_output)

            for name, score in concept_scores:
                try:
                    LLM_theta[stu_idx, theta_id_map[name]] = score
                except:
                    print(name, score)
        return LLM_theta
