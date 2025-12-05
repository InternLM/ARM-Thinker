import io
import ast
import contextlib
import re
import traceback
import multiprocessing
from openai import OpenAI
import httpx
from dotenv import load_dotenv
import os
import base64

# load env
load_dotenv()
# openai_api_key_4o_mini = os.getenv("OPENAI_API_KEY_4O_MINI")
# openai_api_base_4o_mini = os.getenv("OPENAI_API_BASE_4O_MINI")

BENCH_LIST = {
    "reward-bench-2": {
        "path": "/path/to/your/raw_dataset_clean.jsonl"
    },
    "fine_grained_perception": {
        "path": "/path/to/your/fine_grained_perception.jsonl",
        "img_root": "/path/to/your/fork/verl/datasets",
        "identity": "idx",
    },
    "multimodal_long_document": {
        "path": "/path/to/your/multimodal_long_document.jsonl",
        "img_root": "/path/to/your/fork/verl/datasets/MMLongBench-Doc-rm/rag/images",
        "identity": "idx",
    },
    "multimodal_instruction_following": {
        "path": "/path/to/your/multimodal_instruction_following.jsonl",
        "img_root": "/path/to/your/LMUData/images/MM-IFEval",
        "identity": "idx",
    },
}

# judge_client_4o_mini = OpenAI(
#     api_key=openai_api_key_4o_mini,
#     base_url=openai_api_base_4o_mini,
#     http_client=httpx.Client(verify=False),
# )

# FIXED_COT_PROMPT = "**Important Requirement:**\nThe given image is `original_image`. You should always output your reasoning within `<think>...</think>`. After reasoning, either provide the final answer within `<answer>...</answer>` or decide to call a tool within `<tool_call>...</tool_call>`. You are encouraged to call the available tools to assist you with judgment or verification and help to answer the question, and you may call them multiple times if needed. If a tool call fails, you can choose to retry based on the error message, or you can stop tool calling and provide your final answer or judgement. Once no further tool calls are needed, put your final answer or judgment within `<answer>...</answer>`."
FIXED_COT_PROMPT = (
    "**Important Requirement:**\n"
    "The given image is `original_image`. You must output your reasoning inside `<think>...</think>`. "
    "After reasoning, either output the final answer within `<answer>...</answer>` or call a tool within `<tool_call>...</tool_call>`. "
    "You may call tools multiple times across turns to assist with judgment or verification, **but only one tool per turn**. "
    "If a tool call fails, you can retry or stop and give your final answer. "
    "Once no more tool calls are needed, provide your final answer or judgment within `<answer>...</answer>`."
)

FIXED_NO_COT_PROMPT = (
    "**Important Note:**\n"
    "The given image is `original_image`."
)

# actually means no tool call, but still need to output the reasoning
FIXED_COT_PROMPT_DIRECT = (
    "**Important Requirement:**\n"
    "You must output your reasoning inside `<think>...</think>`. "
    "After reasoning, You should output the your final answer or judgment within `<answer>...</answer>`."
)


def _run_code_in_subprocess(code: str, output_queue):
    buffer = io.StringIO()
    try:
        tree = ast.parse(code, mode="exec")
        var_names = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last_expr = tree.body.pop()
            val = last_expr.value
            if isinstance(val, ast.Name):
                var_names = [val.id]
            elif isinstance(val, ast.Tuple):
                names = [elt.id for elt in val.elts if isinstance(elt, ast.Name)]
                if names:
                    var_names = names
            assign = ast.Assign(
                targets=[ast.Name(id="_last", ctx=ast.Store())], value=val
            )
            ast.copy_location(assign, last_expr)
            ast.copy_location(assign.targets[0], last_expr)
            tree.body.append(assign)
        tree = ast.fix_missing_locations(tree)
        mod = compile(tree, filename="<ast>", mode="exec")
        namespace = {}
        with contextlib.redirect_stdout(buffer):
            exec(mod, namespace)
        output = buffer.getvalue().strip()
        if not output and "_last" in namespace:
            last_val = namespace["_last"]
            if var_names:
                if len(var_names) == 1:
                    output = f"{var_names[0]} = {repr(last_val)}"
                else:
                    vals = list(last_val) if isinstance(last_val, tuple) else [last_val]
                    output = ", ".join(
                        f"{n} = {repr(v)}" for n, v in zip(var_names, vals)
                    )
            else:
                output = repr(last_val)
        output_queue.put({"output": output})
    except Exception:
        output_queue.put({"error": traceback.format_exc().strip()})


# run code and return the output
def run_code(code: str, timeout: int = 5) -> str:
    output_queue = multiprocessing.Queue()
    p = multiprocessing.Process(
        target=_run_code_in_subprocess, args=(code, output_queue)
    )
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        return "Error: Code execution timed out."
    if not output_queue.empty():
        result = output_queue.get()
        if "output" in result:
            return result["output"]
        else:
            return f"Error during execution:\n{result['error']}"
    return "Error: No output captured."


# extract python code from the response
def extract_python_code(res):
    import re

    m = re.search(r"```python(.*?)```", res, re.DOTALL)
    return m.group(1).strip() if m else None


# extract true/false label from the response
def extract_true_false_label(res):
    # Search from the end for true/false wrapped in [[]]
    bracket_matches = re.findall(r"\[\[\s*(true|false)\s*\]\]", res, re.IGNORECASE)

    # Search from the end for true/false wrapped in {}, e.g., \boxed{\text{False}}
    brace_matches = re.findall(r"\{\s*(true|false)\s*\}", res, re.IGNORECASE)

    # Search from the end for bare true/false (case-insensitive)
    bare_matches = re.findall(r"\b(true|false)\b", res, re.IGNORECASE)

    # Merge all matches (priority: [] > {} > bare) and take the last one
    all_matches = bracket_matches + brace_matches + bare_matches

    return all_matches[-1].lower() if all_matches else None


def encode_image_file(image_file):
    with open(image_file, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


EXT_TO_MIME = {
    "JPEG": "image/jpeg",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "jpe": "image/jpeg",
    "png": "image/png",
    "gif": "image/gif",
    "bmp": "image/bmp",
    "webp": "image/webp",
    "tif": "image/tiff",
    "tiff": "image/tiff",
    "ico": "image/x-icon",
    "svg": "image/svg+xml",
    "heic": "image/heic",
    "heif": "image/heif",
}


def message_format_with_image_file(user_prompt, image_file):

    image_type = image_file.split(".")[-1]
    mime_type = EXT_TO_MIME[image_type]
    base64_image = encode_image_file(image_file)
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{base64_image}"
                        # "detail": "low"
                    },
                },
                {"type": "text", "text": user_prompt},
            ],
        },
    ]


# def extract_choice_from_resp_single_rm(resp_content):
#     """
#     Extract True or False from the response content.
#     Support both with <answer> tag and without tag.
#     Return None if not found.
#     """
#     pattern = (
#         r"(?:Overall\s*Judgment\s*[:：\-]?\s*)?(?:The\s*answer\s*is\s*)?(True|False)"
#     )
#     match = re.search(pattern, resp_content, re.IGNORECASE)
#     if match:
#         return match.group(1).capitalize()  # return "True" or "False"
#     return None
def extract_choice_from_resp_single_rm(resp_content):
    # 1. Try to extract from <answer>...</answer>
    m = re.search(
        r"<answer>(.*?)</answer>",
        resp_content,
        re.IGNORECASE | re.DOTALL
    )
    if m:
        inner = m.group(1)
        m2 = re.search(r"\b(True|False)\b", inner, re.IGNORECASE)
        if m2:
            return m2.group(1).capitalize()

    # 2. Fallback: use the original pattern, but match ALL occurrences and take the last one
    pattern = (
        r"(?:Overall\s*Judgment\s*[:：\-]?\s*)?"
        r"(?:The\s*answer\s*is\s*)?"
    )
    matches = re.findall(pattern, resp_content, re.IGNORECASE)

    if matches:
        return matches[-1].capitalize()

    return None


def extract_choice_from_resp_pair_rm(resp_content):
    # try to match the standard judgment statement from the end
    pattern = r"Answer\s*(\d+)\s*is\s*(?:the\s*)?(?:slightly\s*)?(?:better|preferred)"
    matches = list(re.finditer(pattern, resp_content, re.IGNORECASE))
    if matches:
        # get the last match
        return int(matches[-1].group(1))
    # Add process answer like "<answer> 1 </answer>"
    pattern = r"<answer>\s*(\d+)\s*</answer>"
    matches = list(re.finditer(pattern, resp_content, re.IGNORECASE))
    if matches:
        return int(matches[-1].group(1))
    return None


LLM_PARSE_ANSWER_PROMPT = """
You are given a pairwise judgement for two responses. Please return the better response according to the judgement.
Return the Answer X ONLY. e.g., Answer 1 or Answer 2.

Judgement: {judgement}
"""

# LLM_PARSE_ANSWER_PROMPT = '''
# You are given a pairwise judgement for two responses. Please return the better response according to the judgement.
# Return the Answer X ONLY. e.g., Answer 1 or Answer 2.

# Judgement: {judgement}
# '''


def make_prompt_offical(data_obj, random_number):
    answers = (
        [data_obj["response"][0], data_obj["response"][1]]
        if random_number == 0
        else [data_obj["response"][1], data_obj["response"][0]]
    )
    prompt_str = f"""You are a highly capable multimodal AI assistant tasked with evaluating answers to visual questions. Please analyze the following image and question, then determine which of the two provided answers is better.

Question: {data_obj["query"]}

Answer 1: {answers[0]}

Answer 2: {answers[1]}

Please evaluate both answers based on the following criteria:
1. Accuracy: How well does the answer align with the visual information in the image?
2. Completeness: Does the answer fully address all aspects of the question?
3. Clarity: Is the answer easy to understand and well-articulated?
4. Relevance: Does the answer directly relate to the question and the image?

After your evaluation, please:
1. Explain your reasoning for each criterion.
2. Provide an overall judgment on which answer is better (Answer 1 or Answer 2). For example: Overall Judgment: Answer X is better.

Your response should be structured and detailed, demonstrating your understanding of both the visual and textual elements of the task."""
    return prompt_str


def make_prompt_simple(data_obj, random_number):
    answers = (
        [data_obj["response"][0], data_obj["response"][1]]
        if random_number == 0
        else [data_obj["response"][1], data_obj["response"][0]]
    )
    prompt_str = f"""You are a highly capable multimodal AI assistant tasked with evaluating answers to visual questions. Please analyze the following image and question, then determine which of the two provided answers is better.

Question: {data_obj["query"]}

Answer 1: {answers[0]}

Answer 2: {answers[1]}

After your evaluation, please:
1. Explain your reasoning for each criterion.
2. Provide an overall judgment on which answer is better (Answer 1 or Answer 2). For example: Overall Judgment: Answer X is better.

Your response should be structured and detailed, demonstrating your understanding of both the visual and textual elements of the task."""
    return prompt_str


def make_prompt_direct_mmif(data_obj):
    prompt = f"""\
You will receive a response(named as `resp_0`) which follows the user's instruction or requirement to the provided image. Your Task is to judge whether the response satisfies the constraint. If it does, you should mark it as `True`, otherwise `False` for you think the response does not satisfy the constraint.

<start_of_instruction>
{data_obj["instruction"]}
<end_of_instruction>

<start_of_response>
{data_obj["prediction"]}
<end_of_response>

<start_of_constraint>
{data_obj["constraints"][0]["value"]}
<end_of_constraint>

## Output Format (strict)
You should make the final judgment wrapped in <answer></answer> XML tags: <answer>Overall Judgment: True (or False)</answer>
"""
    return prompt


# Add on 2025.09.26
def make_prompt_single_rm_mmif(instruction, prediction, constraints):
    # print(f"constraints: {constraints}")
    # print(f"type of constraints: {type(constraints)}")
    prompt = f"""\
You will receive a response(named as `text_0`) which follows the user's instruction or requirement to the provided image. Your Task is to judge whether the response satisfies the constraint. If it does, you should mark it as `True`, otherwise `False` for you think the response does not satisfy the constraint.

<start_of_instruction>
{instruction}
<end_of_instruction>

<start_of_text_0>
{prediction}
<end_of_text_0>

<start_of_constraint>
{constraints[0]["value"]}
<end_of_constraint>

## Output Format (strict)
You should make the final judgment wrapped in <answer></answer> XML tags: <answer>Overall Judgment: True (or False)</answer>
"""
    return prompt


# Add on 2025.09.22
def make_prompt_single_rm(instruction, prediction):
    prompt = f"""\
You will receive a response(named as `resp_0`) which follows the user's instruction or requirement to the provided image (or document). Your Task is to judge whether the response is correct or not. If it is correct, you should mark it as `True`, otherwise `False` for you think the response is not correct.

<start_of_instruction>
{instruction}
<end_of_instruction>

<start_of_resp_0>
{prediction}
<end_of_resp_0>

## Output Format (strict)
You should make the final judgment wrapped in <answer></answer> XML tags: <answer>Overall Judgment: True (or False)</answer>
"""
    return prompt


# Add on 2025.09.23
def make_prompt_single_rm_without_img(instruction, prediction):
    prompt = f"""\
You will receive a response(named as `resp_0`) which answers a question about a document. Your Task is to judge whether the response is correct or not. If it is correct, you should mark it as `True`, otherwise `False` for you think the response is not correct.

<start_of_instruction>
{instruction}
<end_of_instruction>

<start_of_resp_0>
{prediction}
<end_of_resp_0>

## Output Format (strict)
You should make the final judgment wrapped in <answer></answer> XML tags: <answer>Overall Judgment: True (or False)</answer>
"""
    return prompt


# Add on 2025.09.22
# def make_prompt_simple(data_obj, random_number):
#     answers = [data_obj["response"][0], data_obj["response"][1]] if random_number == 0 else [data_obj["response"][1], data_obj["response"][0]]
#     prompt_str = f'''You are a highly capable multimodal AI assistant tasked with evaluating answers to visual questions. Please analyze the following image and question, then determine which of the two provided answers is better.

# Question: {data_obj["query"]}

# Answer 1: {answers[0]}

# Answer 2: {answers[1]}

# After your evaluation, please:
# 1. Explain your reasoning for each criterion.
# 2. Provide an overall judgment on which answer is better (Answer 1 or Answer 2). For example: Overall Judgment: Answer X is better.


# Your response should be structured and detailed, demonstrating your understanding of both the visual and textual elements of the task.'''
#     return prompt_str


# Add on 2025.09.28
def make_prompt_pair_rm(instruction, prediction1, prediction2):
    prompt = f"""\
You will receive two responses (named as `resp_1` and `resp_2`) which follow the user's instruction or requirement to the provided image (or document). Your Task is to judge which response is better. Note that correctness is most important. If both are not correct, you should choose the one that is more better from other aspects.

<start_of_instruction>
{instruction}
<end_of_instruction>

<start_of_resp_1>
{prediction1}
<end_of_resp_1>

<start_of_resp_2>
{prediction2}
<end_of_resp_2>

## Output Format (strict)
You should make the final judgment wrapped in <answer></answer> XML tags: <answer>Overall Judgment: Answer X is better (X must be either 1 or 2). </answer>
"""
    return prompt

# Add on 2025.11.04
def make_prompt_4_way_pair_rm(instruction, prediction1, prediction2, prediction3, prediction4):
    prompt = f"""\
You will receive four responses (named as `resp_1`, `resp_2`, `resp_3`, `resp_4`) which follow the user's instruction or requirement to the provided image (or document). Your Task is to judge which response is better. Note that correctness is most important. If all are not correct, you should choose the one that is more better from other aspects.

<start_of_instruction>
{instruction}
<end_of_instruction>

<start_of_resp_1>
{prediction1}
<end_of_resp_1>

<start_of_resp_2>
{prediction2}
<end_of_resp_2>

<start_of_resp_3>
{prediction3}
<end_of_resp_3>

<start_of_resp_4>
{prediction4}
<end_of_resp_4>

## Output Format (strict)
You should make the final judgment wrapped in <answer></answer> XML tags: <answer>Overall Judgment: Answer X is better (X must be either 1, 2, 3, or 4). </answer>
"""
    return prompt


def make_prompt_agent_pair_vl_reward_bench(data_obj, random_number):
    answers = (
        [data_obj["response"][0], data_obj["response"][1]]
        if random_number == 0
        else [data_obj["response"][1], data_obj["response"][0]]
    )

    prompt = make_prompt_pair_rm(
        data_obj.get("instruction", data_obj["query"]), answers[0], answers[1]
    )
    return prompt


def make_prompt_agent_mmif(data_obj):
    prompt = f"""\
You will receive a response(named as `text_0`) which follows the user's instruction to the provided image. Your Task is to judge whether the response satisfies the constraint. If it does, you should mark it as `True`, otherwise `False` for you think the response does not satisfy the constraint.

<start_of_instruction>
{data_obj["instruction"]}
<end_of_instruction>

<start_of_response>
{data_obj["prediction"]}
<end_of_response>

<start_of_constraint>
{data_obj["constraints"][0]["value"]}
<end_of_constraint>

## Output Format (strict)
At each step, your output must be **exactly one** of the following:
1. If you find a tool call is needed so you call a tool in the end warpped in <tool_call></tool_call> XML tags (You can call tools multiple times but only one tool call at a time): <tool_call>...</tool_call>
2. If all needed tools have been used and you are ready to conclude, you should make the final judgment wrapped in <answer></answer> XML tags: <answer>Overall Judgment: True (or False)</answer>
"""
    return prompt


def make_prompt_agent_mmif_verl(data_obj):
    prompt = f"""\
You will receive a response(named as `text_0`) which follows the user's instruction to the provided image. Your Task is to judge whether the response satisfies the constraint. If it does, you should mark it as `True`, otherwise `False` for you think the response does not satisfy the constraint.

<start_of_instruction>
{data_obj["instruction"]}
<end_of_instruction>

<start_of_response>
{data_obj["prediction"]}
<end_of_response>

<start_of_constraint>
{data_obj["constraints"][0]["value"]}
<end_of_constraint>

## Tool Calling Important Notes:
- When calling tools, **DO NOT** pass the original response text as arguments
- Instead, use the reference identifier `text_0` to refer to the response
- This helps avoid token limit issues and makes the tool calls more efficient
- Example: Use `"response": "text_0"` instead of copying the entire response text

## Output Format (strict)
At each step, your output must be **exactly one** of the following:
1. If you find a tool call is needed so you call a tool in the end warpped in <tool_call></tool_call> XML tags (You can call tools multiple times but only one tool call at a time): <tool_call>...</tool_call>
2. If all needed tools have been used and you are ready to conclude, you should make the final judgment wrapped in <answer></answer> XML tags: <answer>Overall Judgment: True (or False)</answer>
"""
    return prompt


# def make_prompt_agent_with_tool(data_obj, random_number):
#     answers = [data_obj["response"][0], data_obj["response"][1]] if random_number == 0 else [data_obj["response"][1], data_obj["response"][0]]

#     prompt_str = f"""\
# You are a capable multimodal AI assistant with access to powerful tools. Your task is to compare two answers to a visual question and decide which is better. Use tools when needed to check facts, compute values, or verify any claims. Be objective and rigorous in your judgment.

# ---

# **Question**: {data_obj["query"]}

# **Answer 1**: {answers[0]}

# **Answer 2**: {answers[1]}

# ---

# Please evaluate the two answers using the following process:

# ### Step 1: Understand the Question
# - Identify the key visual or conceptual elements that a correct answer must address.

# ### Step 2: Evaluate Each Answer
# Assess each answer independently across the four criteria:
# 1. **Accuracy** – Is the content factually and visually correct?
# 2. **Completeness** – Does it fully address all aspects of the question?
# 3. **Clarity** – Is it well-structured, coherent, and easy to understand?
# 4. **Relevance** – Does it stay focused and avoid unnecessary information?

# If you encounter any claim that seems uncertain or possibly incorrect, **explain your reasoning and use the appropriate tool** to verify:
# - Use `web_search` for external facts (e.g., dates, entities).
# - Use `python_executor` for logical reasoning or computations.

# You may call tools **multiple times** if needed. Do not assume correctness without verification.

# ### Step 3: Compare and Decide
# - Summarize the strengths and weaknesses of both answers based on the above.
# - Make a justified, concise final judgment.

# ---

# **Your final output must include**:
# 1. Your full reasoning and comparisons.
# 2. A final line: `Overall Judgment: Answer X is better` (X is 1 or 2).
# """
#     return prompt_str


# def make_prompt_agent_with_tool(data_obj, random_number):
#     answers = (
#         [data_obj["response"][0], data_obj["response"][1]]
#         if random_number == 0
#         else [data_obj["response"][1], data_obj["response"][0]]
#     )

#     prompt_str = f"""\
# You are a capable multimodal agent with access to several useful tools (`web_search`, `python_executor`). Your task is to compare two answers to a visual question and determine which one is better.

# Instructions:
# 1. Understand the question and identify key visual or conceptual elements that a correct answer must address.
# 2. Evaluate both answers along these four criteria:
#    - **Accuracy** – Are the facts and visual interpretations correct?
#    - **Completeness** – Does the answer address all parts of the question?
#    - **Clarity** – Is it clear and easy to follow?
#    - **Relevance** – Does it stay on topic without extra fluff?
# 3. For **any uncertain or detailed claim**, you **must attempt verification** using tools:
#    - Use `web_search` for facts, dates, named entities, etc.
#    - Use `python_executor` for logic, math, or visual reasoning.
# 4. You can call tools **multiple times**. Do **not assume correctness**—verify when in doubt.

# ---

# Question: {data_obj["query"]}

# Answer 1: {answers[0]}

# Answer 2: {answers[1]}

# Now, reason step by step and use tools when needed. Be rigorous and unbiased.

# When you are able to make a final judgment, output in this format:
# 1. Your full reasoning and comparisons.
# 2. A final line in this exact format: `Overall Judgment: Answer X is better` (X is 1 or 2).
# """
#     return prompt_str


# def make_prompt_agent_with_tool_v4(data_obj, random_number):
#     answers = (
#         [data_obj["response"][0], data_obj["response"][1]]
#         if random_number == 0
#         else [data_obj["response"][1], data_obj["response"][0]]
#     )

#     prompt_str = f"""\
# Your task is to **compare two answers** of a visual question to the image which I provide and determine which one is better. **If both answers are flawed, you must choose the one that better satisfies the criteria overall.**

# # Output Format (strict)
# At each step, your output must be **exactly one** of the following:
# 1. First reason or continue reasoning and you find a tool call is needed so you call a tool in the end warpped in <tool_call></tool_call> XML tags (You can call tools multiple times but only one tool call at a time): ...<tool_call>...</tool_call>
# 2. If all needed tools have been used and you are ready to conclude, you should first reason and then end with a final judgment wrapped in <answer></answer> XML tags: ...<answer>Overall Judgment: Answer X is better(X must be either 1 or 2)</answer>

# # Evaluation Instructions
# 1. Carefully analyze the image, the visual question and both answers.
# 2. Evaluate both answers based on:
#    * Accuracy (Are claims and observations correct?)
#    * Completeness (Are all aspects of the question addressed?)
#    * Clarity (Is the explanation understandable?)
#    * Relevance (Is it focused, with no unnecessary info?)
# 4. Use tools whenever you think you need and correctly use them.

# # Tool Usage Instructions
# 1. **web_search**: Use it to look up facts, dates, names, and other text-based information. Do not use web_search to answer questions about the content of an image (e.g. location, color, actions) like "details about the train and buildings in the image" or "person in the image". Instead, extract key details or make an assumption based on the image first, then search using relevant keywords to get the answer or verify your assumption.

# ---
# Now, here is the question and two answers, and **you are strictly required to make a choice between Answer 1 and Answer 2 even you think both are flawed or good enough**:

# Question: {data_obj["query"]}

# Answer 1: {answers[0]}

# Answer 2: {answers[1]}
# """
#     return prompt_str


# def make_prompt_agent_with_crop_search_at_most_twice(data_obj, random_number):
#     answers = (
#         [data_obj["response"][0], data_obj["response"][1]]
#         if random_number == 0
#         else [data_obj["response"][1], data_obj["response"][0]]
#     )

#     prompt_str = f"""\
# Your task is to **compare two answers** of a visual question to the image which I provide (named as `image_0`) and determine which one is better. **If both answers are flawed, you must choose the one that better satisfies the criteria overall.**

# # Output Format (strict)
# At each step, your output must be **exactly one** of the following:
# 1. First reason or continue reasoning and you find a tool call is needed so you call a tool in the end warpped in <tool_call></tool_call> XML tags (You can call tools multiple times but only one tool call at a time): ...<tool_call>...</tool_call>
# 2. If all needed tools have been used and you are ready to conclude, you should first reason and then end with a final judgment wrapped in <answer></answer> XML tags: ...<answer>Overall Judgment: Answer X is better(X must be either 1 or 2)</answer>

# # Evaluation Instructions
# 1. Carefully analyze the image, the visual question and both answers.
# 2. Use tools whenever you think you need and correctly use them.
# 3. For `crop_image` tool, you can call it at most twice. This tool is for you to zoom in some specific area of the image.
# 4. for `web_search` tool, you can call it at most once. This tool is for you to search some knowledge that from internet, like "the height of the Eiffel Tower", and bad requests like "color in the image" is not allowed (because this is not a knowledge question and the answer will not be found from internet)
# 5. You should not call tools when you don't need them.

# ---
# Now, here is the question and two answers, and **you are strictly required to make a choice between Answer 1 and Answer 2 even you think both are flawed or good enough**:

# Question: {data_obj["query"]}

# Answer 1: {answers[0]}

# Answer 2: {answers[1]}
# """
#     return prompt_str


# def make_prompt_agent_with_crop_at_most_twice(data_obj, random_number):
#     answers = (
#         [data_obj["response"][0], data_obj["response"][1]]
#         if random_number == 0
#         else [data_obj["response"][1], data_obj["response"][0]]
#     )

#     prompt_str = f"""\
# Your task is to **compare two answers** of a visual question to the image which I provide (named as `image_0`) and determine which one is better. **If both answers are flawed, you must choose the one that better satisfies the criteria overall.**

# # Output Format (strict)
# At each step, your output must be **exactly one** of the following:
# 1. First reason or continue reasoning and you find a tool call is needed so you call a tool in the end warpped in <tool_call></tool_call> XML tags (You can call tools multiple times but only one tool call at a time): ...<tool_call>...</tool_call>
# 2. If all needed tools have been used and you are ready to conclude, you should first reason and then end with a final judgment wrapped in <answer></answer> XML tags: ...<answer>Overall Judgment: Answer X is better(X must be either 1 or 2)</answer>

# # Evaluation Instructions
# 1. Carefully analyze the image, the visual question and both answers.
# 2. Use tools whenever you think you need and correctly use them.
# 3. For `crop_image` tool, you can call it at most twice. This tool is for you to zoom in some specific area of the image.
# 4. You should not call tools when you don't need them.

# ---
# Now, here is the question and two answers, and **you are strictly required to make a choice between Answer 1 and Answer 2 even you think both are flawed or good enough**:

# Question: {data_obj["query"]}

# Answer 1: {answers[0]}

# Answer 2: {answers[1]}
# """
#     return prompt_str


#     prompt_str = f"""\
# You are a careful and capable multimodal agent.
# Your task is to **compare two answers** to a visual question and determine which one is better based on the image.

# # Task Rules
# You must choose one of the two answers. No ties, no ambiguity — pick the better one, even if both are flawed.

# # Step-by-Step Output (STRICT FORMAT)
# You must reason step by step. At each step, choose **exactly one** of:

# ## 1. If you need external information:
# <think>Explain your reasoning and why you now need a tool.</think>
# <tool_call>tool_name[query or code]</tool_call>

# ## 2. If you're ready to make a final judgment:
# <think>Summarize your reasoning and state which answer is better.</think>
# <answer>Overall Judgment: Answer X is better</answer>

# (X must be either 1 or 2)

# # Evaluation Criteria
# Compare the answers on:
# - **Accuracy**: Are the claims correct and supported by the image?
# - **Completeness**: Does the answer address the full question?
# - **Clarity**: Is it easy to understand?
# - **Relevance**: Does it stay on topic?

# # Tool Usage Policy (USE ONLY IF NECESSARY)
# You may use the following tools **only if you cannot verify something from the image alone**:

# ## web_search[query]
# - Use: real-world facts, entity info, name definitions
#   - Example: "Who is Stan Doe?"
#   - Example: "When was the Eiffel Tower built?"
# - DO NOT use for:
#   - "What color is the shirt in the image?"
#   - "What is the person doing?" → Use your own visual reasoning!

# ## python_executor[code]
# - Use this for counting, comparing numbers, or visual structure reasoning.
# - Always include `print(...)` to see results.

# # If you're unsure — default to **your own multimodal judgment** unless the answer requires factual verification.

# ---

# Now, here is the visual question and two answers:

# Question: {data_obj["query"]}
# Answer 1: {answers[0]}
# Answer 2: {answers[1]}

# """
#     return prompt_str
