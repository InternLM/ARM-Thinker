import base64
import csv
import io
import contextlib
import logging
import multiprocessing
import re
import sys
import threading
import traceback
import ast
import json
import os
import argparse
import httpx
import numpy as np
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime as dt
    
from eval_utils.utils import BENCH_LIST, message_format_with_image_file, extract_choice_from_resp_single_rm, extract_choice_from_resp_pair_rm
from eval_utils.utils import (
    make_prompt_single_rm,
    make_prompt_pair_rm,
    make_prompt_single_rm_mmif,
    FIXED_COT_PROMPT,
    FIXED_NO_COT_PROMPT,
    make_prompt_4_way_pair_rm
)
from arm_agent.agent_verl import VerlAgent

COT_TOOL_CALL = os.environ.get("COT_TOOL_CALL", "1")
if COT_TOOL_CALL == "1":
    COT_TOOL_CALL = True
else:
    COT_TOOL_CALL = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)

# --- Argument parsing for worker count ---
parser = argparse.ArgumentParser(
    description="Parallel evaluation of LLM math verifications"
)
parser.add_argument("--run_name", type=str, default="", help="Run name (default: )")
parser.add_argument(
    "--model", type=str, default="gpt-4.1", help="Model name (default: gpt-4.1)"
)
parser.add_argument(
    "--workers",
    type=int,
    default=8,
    help="Number of threads to use (default: ThreadPoolExecutor default)",
)
# default is jiaqi boyue key, not used in this script
parser.add_argument(
    "--api_base",
    type=str,
    default="http://35.220.164.252:3888/v1/",
    help="Model name (default: gpt-4.1)",
)
parser.add_argument(
    "--api_key",
    type=str,
    default="sk-5HPKkzKM8QsoipDu2FTGTeATeNoTRRXe6q18hUtAVSJcGmCr",
    help="Model name (default: gpt-4.1)",
)
parser.add_argument(
    "--proxy", type=bool, default=False, help="Use proxy (default: False)"
)
parser.add_argument(
    "--bench_name", type=str, default="", help="Use data (default: False)"
)
parser.add_argument(
    "--work_dir",
    type=str,
    default="./results",
    help="Work directory (default: ./results)",
)
parser.add_argument(
    "--temperature", type=float, default=0.0, help="Temperature (default: 0.0)"
)
parser.add_argument(
    "--max_tokens", type=int, default=2048, help="Max tokens (default: 2048)"
)
parser.add_argument(
    "--tool_config_path", type=str, default="", help="Tool config path (default: )"
)
parser.add_argument("--max_round", type=int, default=10, help="Max round (default: 10)")
parser.add_argument(
    "--max_tool_response_length",
    type=int,
    default=2048,
    help="Max tool response length (default: 2048)",
)

args = parser.parse_args()

# global variables
bench_identity = ""
save_dir = ""

# proxy setting
if not args.proxy:
    os.environ.update(
        {"http_proxy": "", "https_proxy": "", "HTTP_PROXY": "", "HTTPS_PROXY": ""}
    )
    logger.info("No proxy")
else:
    os.environ.update(
        {
            "http_proxy": "http://127.0.0.1:7890",
            "https_proxy": "http://127.0.0.1:7890",
            "HTTP_PROXY": "http://127.0.0.1:7890",
            "HTTPS_PROXY": "http://127.0.0.1:7890",
        }
    )
    logger.info("Use proxy 127.0.0.1:7890")

# temp dir for tool call images
timestamp = dt.datetime.now().strftime("%Y%m%d%H%M%S")
temp_dir = f"/tmp/agent_images_{timestamp}"
os.environ["TOOL_CALL_IMG_TEMP"] = temp_dir
os.makedirs(temp_dir, exist_ok=True)

# model setting
agent = VerlAgent(
    api_base=args.api_base,
    api_key=args.api_key,
    tool_config_path=args.tool_config_path,
    temperature=args.temperature,
    max_tokens=args.max_tokens,
    model_name=args.model,
    max_round=args.max_round,
    max_tool_response_length=args.max_tool_response_length,
)

def run_once_with_prompt_agent(messages, text_0=None):
    rtn = agent.run(messages, text_0=text_0)
    return rtn


def run_once_with_prompt_single_turn(
    model_client,
    model_name,
    messages,
    retry=3,
    temperature=0.0,
    max_tokens=2048,
    n=1,
):
    num_retries = 0

    while num_retries < retry:
        try:
            chat_response = model_client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                n=n,
                # extra_body={"chat_template_kwargs": {"enable_thinking": True}},
            )
            return chat_response
        except Exception as e:
            logger.error(
                f"[Retry {num_retries+1}/{retry}] Exception type: {type(e).__name__}"
            )
            logger.error(f"Error message: {e}")
            logger.error("Traceback:\n" + traceback.format_exc())
            num_retries += 1
    raise RuntimeError(f"Calling OpenAI API failed after {retry} retries.")


    # fallback - call llm to parse the response
    # extract_prompt = LLM_PARSE_ANSWER_PROMPT.format(judgement=resp_content)
    # messages = [
    #     {
    #         "role": "user",
    #         "content": extract_prompt
    #     }
    # ]
    # resp = run_once_with_prompt_single_turn(
    #     model_client=judge_client_4o_mini,
    #     model_name="gpt-4o-mini",
    #     messages=messages,
    #     temperature=0.3,
    #     max_tokens=32,
    #     n=1
    # )
    # resp_content = resp.choices[0].message.content
    # logger.info(f"extract_resp:\n{resp_content}")
    # resp_extract_content = resp_content.strip().lower()

    # # more flexible regex matching llm output (compatible with concise statements)
    # fallback_pattern = r"answer\s*(\d+)(?:\s*is\s*(?:the\s*)?(?:slightly\s*)?(?:better|preferred))?"
    # fallback_matches = list(re.finditer(fallback_pattern, resp_extract_content, re.IGNORECASE))
    # if fallback_matches:
    #     return int(fallback_matches[-1].group(1))  # get the last Answer X

    # all failed, return -1
    # return -1


# process both "single_rm" and "pair_rm"
def process_item(item):
    judge_result = {}

    try:
        if item["rm_tag"] == "single_rm":
            if "mmif" in args.bench_name.lower():
                question = make_prompt_single_rm_mmif(item["question"], item["preds"][0], item["constraints"])
            else:
                question = make_prompt_single_rm(item["question"], item["preds"][0])
            # Delete possible <image> tag, because vllm server does not need it
            # See https://vllm.hyper.ai/docs/inference-and-serving/multimodal_inputs#%E5%9B%BE%E5%83%8F%E8%BE%93%E5%85%A5-1
            question = question.replace("<image>\n", "")
            question = question.replace("<image>", "")
            if "doc_id" in item.keys():
                doc_id_rm_pdf = item["doc_id"].split(".pdf")[0]
                doc_id_prompt = f"The given document is named `{doc_id_rm_pdf}`. The page indices in the combined image start from 1 at the top-left corner and increase horizontally from left to right, then continue to the next row from top to bottom."
                if COT_TOOL_CALL == True:
                    question = question + "\n\n" + FIXED_COT_PROMPT.replace("The given image is `original_image`.", doc_id_prompt)
                else:
                    question = question + "\n\n" + FIXED_NO_COT_PROMPT.replace("The given image is `original_image`.", doc_id_prompt)
                # print(f"DOC_ID_PROMPT: {doc_id_prompt}")
            else:
                if COT_TOOL_CALL == True:
                    question = question + "\n\n" + FIXED_COT_PROMPT
                else:
                    question = question + "\n\n" + FIXED_NO_COT_PROMPT
            real_image_path = os.path.join(bench_img_root, item["images"][0])
            messages = message_format_with_image_file(question, real_image_path)
            resp, tool_call_cnt = run_once_with_prompt_agent(messages, text_0=item["preds"][0])
            # for msg in resp:
            #     content = msg.get("content", "")
            #     # ignore image part, also can choose to save it in img dir
            #     # if role == "user" and isinstance(content, list):
            #     if isinstance(content, list):
            #         for it in content:
            #             # print(f"it: {it}")
            #             if "type" in it and it["type"] == "image_url":
            #                 it["image_url"]["url"] = "hidden"
            # judge_result["full_messages"] = resp
            judge_result["tool_call_cnt"] = tool_call_cnt
            logger.info(f"tool_call_cnt: {tool_call_cnt}")
            resp_content = resp[-1]["content"]
            logger.info(f"resp_content:\n{resp_content}")
            # None, True, False
            model_choice = extract_choice_from_resp_single_rm(resp_content)
        elif item["rm_tag"] == "pair_rm" or item["rm_tag"] == "2-way-pair_rm":
            question = make_prompt_pair_rm(
                item["question"], item["preds"][0], item["preds"][1]
            )
            question = question.replace("<image>\n", "")
            question = question.replace("<image>", "")
            real_image_path = os.path.join(bench_img_root, item["images"][0])
            if "doc_id" in item.keys():
                doc_id_rm_pdf = item["doc_id"].split(".pdf")[0]
                doc_id_prompt = f"The given document is named `{doc_id_rm_pdf}`. The page indices in the combined image start from 1 at the top-left corner and increase horizontally from left to right, then continue to the next row from top to bottom."
                if COT_TOOL_CALL == True:
                    question = question + "\n\n" + FIXED_COT_PROMPT.replace("The given image is `original_image`.", doc_id_prompt)
                else:
                    question = question + "\n\n" + FIXED_NO_COT_PROMPT.replace("The given image is `original_image`.", doc_id_prompt)
                # print(f"DOC_ID_PROMPT: {doc_id_prompt}")
            else:
                if COT_TOOL_CALL == True:
                    question = question + "\n\n" + FIXED_COT_PROMPT
                else:
                    question = question + "\n\n" + FIXED_NO_COT_PROMPT
            messages = message_format_with_image_file(question, real_image_path)
            # TODO: mmif need to add text_0 and text_1 maybe
            resp, tool_call_cnt = run_once_with_prompt_agent(messages)
            # for msg in resp:
            #     content = msg.get("content", "")
            #     # ignore image part, also can choose to save it in img dir
            #     # if role == "user" and isinstance(content, list):
            #     if isinstance(content, list):
            #         for it in content:
            #             # print(f"item: {item}")
            #             if "type" in it and it["type"] == "image_url":
            #                 it["image_url"]["url"] = "hidden"
            # judge_result["full_messages"] = resp
            judge_result["tool_call_cnt"] = tool_call_cnt
            logger.info(f"tool_call_cnt: {tool_call_cnt}")
            resp_content = resp[-1]["content"]
            logger.info(f"resp_content:\n{resp_content}")
            # None, 1, 2, 3, 4
            model_choice = extract_choice_from_resp_pair_rm(resp_content)
        elif item["rm_tag"] == "4-way-pair_rm":
            question = make_prompt_4_way_pair_rm(
                item["question"], item["preds"][0], item["preds"][1], item["preds"][2], item["preds"][3]
            )
            question = question.replace("<image>\n", "")
            question = question.replace("<image>", "")
            real_image_path = os.path.join(bench_img_root, item["images"][0])
            if "doc_id" in item.keys():
                doc_id_rm_pdf = item["doc_id"].split(".pdf")[0]
                doc_id_prompt = f"The given document is named `{doc_id_rm_pdf}`. The page indices in the combined image start from 1 at the top-left corner and increase horizontally from left to right, then continue to the next row from top to bottom."
                if COT_TOOL_CALL == True:
                    question = question + "\n\n" + FIXED_COT_PROMPT.replace("The given image is `original_image`.", doc_id_prompt)
                else:
                    question = question + "\n\n" + FIXED_NO_COT_PROMPT.replace("The given image is `original_image`.", doc_id_prompt)
                # print(f"DOC_ID_PROMPT: {doc_id_prompt}")
            else:
                if COT_TOOL_CALL == True:
                    question = question + "\n\n" + FIXED_COT_PROMPT
                else:
                    question = question + "\n\n" + FIXED_NO_COT_PROMPT
            messages = message_format_with_image_file(question, real_image_path)
            # TODO: mmif need to add text_0 and text_1 maybe
            resp, tool_call_cnt = run_once_with_prompt_agent(messages)
            # for msg in resp:
            #     content = msg.get("content", "")
            #     # ignore image part, also can choose to save it in img dir
            #     # if role == "user" and isinstance(content, list):
            #     if isinstance(content, list):
            #         for it in content:
            #             # print(f"item: {item}")
            #             if "type" in it and it["type"] == "image_url":
            #                 it["image_url"]["url"] = "hidden"
            # judge_result["full_messages"] = resp
            judge_result["tool_call_cnt"] = tool_call_cnt
            logger.info(f"tool_call_cnt: {tool_call_cnt}")
            resp_content = resp[-1]["content"]
            logger.info(f"resp_content:\n{resp_content}")
            # None, 1, 2, 3, 4
            model_choice = extract_choice_from_resp_pair_rm(resp_content)
        else:
            raise ValueError(f"Unsupported rm_tag: {item['rm_tag']}")
        idxxx = item[bench_identity]
        img_save_root = os.path.join(save_dir, "images")
        save_img_root = os.path.join(img_save_root, str(idxxx))
        os.makedirs(save_img_root, exist_ok=True)
        img_counter = 0
        for msg in resp:
            content = msg.get("content", "")
            if not isinstance(content, list):
                continue
            for itemm in content:
                if not (isinstance(itemm, dict) and itemm.get("type") == "image_url"):
                    continue
                image_url = itemm.get("image_url", {}).get("url", "")
                if not image_url:
                    continue
                try:
                    # judge whether it is base64 or url
                    if image_url.startswith("data:image"):
                        header, b64_data = image_url.split(",", 1)
                        ext = header.split("/")[1].split(";")[0]  # extract extension
                        img_bytes = base64.b64decode(b64_data)
                        img_name = f"img_{img_counter:04d}.{ext}"
                    else:
                        raise ValueError(
                            f"Invalid image url: {image_url}, can only process base64 image url"
                        )
                    # save image
                    img_path = os.path.join(save_img_root, img_name)
                    with open(img_path, "wb") as f:
                        f.write(img_bytes)

                    # replace image_url["url"] with relative path
                    itemm["image_url"]["url"] = f"{str(idxxx)}/{img_name}"
                    img_counter += 1

                    print(f"[Saved] {img_path}, {itemm['image_url']['url']}")
                except Exception as e:
                    logger.error(f"Error: {e}")
                    logger.error("Traceback:\n" + traceback.format_exc())
                    raise
        # for msg in resp:
        #     content = msg.get("content", "")
        #     # ignore image part, also can choose to save it in img dir
        #     # if role == "user" and isinstance(content, list):
        #     if isinstance(content, list):
        #         for it in content:
        #             # print(f"item: {item}")
        #             if "type" in it and it["type"] == "image_url":
        #                 it["image_url"]["url"] = "hidden"
        judge_result["full_messages"] = resp
        gt = item["gt"]
        judge_result["model_resp"] = resp_content
        judge_result["model_choice"] = model_choice
        judge_result["gt"] = gt
        judge_result["score"] = 1 if model_choice == gt else 0
        if model_choice is None:
            judge_result["flag_status"] = "ExtractFailed"
        else:
            judge_result["flag_status"] = "ExtractSuccess"
        return judge_result
    except Exception as e:
        logger.error(f"Error: {e}")
        logger.error("Traceback:\n" + traceback.format_exc())
        raise


# Main execution
if __name__ == "__main__":
    # >>> inference <<<
    bench_name = args.bench_name
    bench_path = BENCH_LIST[bench_name]["path"]
    bench_img_root = BENCH_LIST[bench_name]["img_root"]
    bench_identity = BENCH_LIST[bench_name]["identity"]
    model_name_in_path = args.model.split("/")[-1]
    if bench_path.endswith(".jsonl"):
        with open(bench_path, "r") as f:
            bench_data = [json.loads(line) for line in f]
    else:
        raise ValueError(f"Currently unsupported file format: {bench_path}")

    worker_num = args.workers
    # save inference results
    save_dir = f"{args.work_dir}/{args.run_name}"
    save_path = f"{save_dir}/{model_name_in_path}.jsonl"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # find missing ids/idxs
    # idx because original id has duplicates, add column "idx"
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            results = [json.loads(line) for line in f]
        missing_idxs = [
            item[bench_identity]
            for item in bench_data
            if item[bench_identity] not in [result[bench_identity] for result in results]
        ]
    else:
        missing_idxs = [item[bench_identity] for item in bench_data]
        results = []
    logger.info(f"Total {len(bench_data)} items, {len(missing_idxs)} items missing.")
    bench_data = [item for item in bench_data if item[bench_identity] in missing_idxs]
    logger.info(f"Evaluating left {len(bench_data)} items...")
    # breakpoint()

    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=worker_num) as executor, open(
        save_path, "a", encoding="utf-8"
    ) as f:
        futures = {executor.submit(process_item, item): item for item in bench_data}
        for future in tqdm(
            as_completed(futures), total=len(bench_data), desc="Evaluating..."
        ):
            try:
                result = future.result()
                if result is not None:
                    item = futures[future]
                    item["judge_result"] = result
                    results.append(item)

                    # Safe write
                    with lock:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
                        f.flush()

            except Exception as e:
                logger.error(f"Error: {e}")
                os._exit(1)

    # >>> scoring: compute average score and save results <<<
    from collections import defaultdict

    # Score grouped by rm_tag
    score_dict = defaultdict(list)
    for item in results:
        score_dict[item["rm_tag"]].append(item["judge_result"]["score"])


    # single_rm
    single_rm_score = sum(score_dict["single_rm"])
    single_rm_count = len(score_dict["single_rm"])
    single_rm_average_score = (
        single_rm_score / single_rm_count if single_rm_count > 0 else 0.0
    )
    logger.info(f"Single RM average score: {single_rm_average_score:.4f}")
    single_rm_score_json = {
        "single_rm_score": single_rm_score,
        "single_rm_count": single_rm_count,
        "single_rm_average_score": single_rm_average_score,
    }

    # pair_rm
    pair_rm_score = sum(score_dict["pair_rm"])
    pair_rm_count = len(score_dict["pair_rm"])
    pair_rm_average_score = pair_rm_score / pair_rm_count if pair_rm_count > 0 else 0.0
    logger.info(f"Pair RM average score: {pair_rm_average_score:.4f}")
    pair_rm_score_json = {
        "pair_rm_score": pair_rm_score,
        "pair_rm_count": pair_rm_count,
        "pair_rm_average_score": pair_rm_average_score,
    }

    # 2-way-pair_rm
    two_way_pair_rm_score = sum(score_dict["2-way-pair_rm"])
    two_way_pair_rm_count = len(score_dict["2-way-pair_rm"])
    two_way_pair_rm_average_score = two_way_pair_rm_score / two_way_pair_rm_count if two_way_pair_rm_count > 0 else 0.0
    logger.info(f"2-way-pair RM average score: {two_way_pair_rm_average_score:.4f}")
    two_way_pair_rm_score_json = {
        "two_way_pair_rm_score": two_way_pair_rm_score,
        "two_way_pair_rm_count": two_way_pair_rm_count,
        "two_way_pair_rm_average_score": two_way_pair_rm_average_score,
    }

    # 4-way-pair_rm
    four_way_pair_rm_score = sum(score_dict["4-way-pair_rm"])
    four_way_pair_rm_count = len(score_dict["4-way-pair_rm"])
    four_way_pair_rm_average_score = four_way_pair_rm_score / four_way_pair_rm_count if four_way_pair_rm_count > 0 else 0.0
    logger.info(f"4-way-pair RM average score: {four_way_pair_rm_average_score:.4f}")
    four_way_pair_rm_score_json = {
        "four_way_pair_rm_score": four_way_pair_rm_score,
        "four_way_pair_rm_count": four_way_pair_rm_count,
        "four_way_pair_rm_average_score": four_way_pair_rm_average_score,
    }

    # overall
    all_scores = [item["judge_result"]["score"] for item in results]
    overall_score = sum(all_scores)
    overall_count = len(all_scores)
    overall_average_score = overall_score / overall_count if overall_count > 0 else 0.0
    logger.info(f"Overall average score: {overall_average_score:.4f}")
    overall_score_json = {
        "overall_score": overall_score,
        "overall_count": overall_count,
        "overall_average_score": overall_average_score,
    }

    # save
    score_json = {
        "single_rm": single_rm_score_json,
        "pair_rm": pair_rm_score_json,
        "2-way-pair_rm": two_way_pair_rm_score_json,
        "4-way-pair_rm": four_way_pair_rm_score_json,
        "overall": overall_score_json,
    }

    with open(
        f"{save_dir}/{model_name_in_path}_scores.json", "w", encoding="utf-8"
    ) as f:
        json.dump(score_json, f, ensure_ascii=False, indent=2)
    logger.info(f"Scores saved to: {f.name}")
    logger.info(f"score_json:\n{json.dumps(score_json, ensure_ascii=False, indent=2)}")
