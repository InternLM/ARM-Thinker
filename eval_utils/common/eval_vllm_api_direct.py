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

from eval_utils.utils import BENCH_LIST, message_format_with_image_file, extract_choice_from_resp_single_rm, extract_choice_from_resp_pair_rm
from eval_utils.utils import (
    make_prompt_single_rm,
    make_prompt_pair_rm,
    make_prompt_single_rm_mmif,
    make_prompt_4_way_pair_rm,
    FIXED_COT_PROMPT_DIRECT,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)

# GLOBAL PARAMS

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

# params related to bench
# score_w_ratings for non-tie subset
# parser.add_argument(
#     "--score_w_ratings",
#     type=bool,
#     default=False,
#     help="Use score_w_ratings (default: False)",
# )

args = parser.parse_args()


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

# model setting
MODEL_NAME = args.model
openai_api_base = args.api_base
openai_api_key = args.api_key
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
    http_client=httpx.Client(verify=False),
)


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

USE_COT = os.environ.get("USE_COT", "0")
if USE_COT == "1":
    USE_COT = True
    print(f"USE_COT: {USE_COT}")
else:
    USE_COT = False
    print(f"USE_COT: {USE_COT}")

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
            if USE_COT == True:
                question += "\n\n" + FIXED_COT_PROMPT_DIRECT
            real_image_path = os.path.join(bench_img_root, item["images"][0])
            messages = message_format_with_image_file(question, real_image_path)
            resp = run_once_with_prompt_single_turn(
                model_client=client,
                model_name=MODEL_NAME,
                messages=messages,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            resp_content = resp.choices[0].message.content
            logger.info(f"resp_content:\n{resp_content}")
            # True, False
            model_choice = extract_choice_from_resp_single_rm(resp_content)
        elif item["rm_tag"] == "pair_rm" or item["rm_tag"] == "2-way-pair_rm":
            question = make_prompt_pair_rm(
                item["question"], item["preds"][0], item["preds"][1]
            )
            question = question.replace("<image>\n", "")
            question = question.replace("<image>", "")
            if USE_COT == True:
                question += "\n\n" + FIXED_COT_PROMPT_DIRECT
            real_image_path = os.path.join(bench_img_root, item["images"][0])
            messages = message_format_with_image_file(question, real_image_path)
            resp = run_once_with_prompt_single_turn(
                model_client=client,
                model_name=MODEL_NAME,
                messages=messages,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            resp_content = resp.choices[0].message.content
            logger.info(f"resp_content:\n{resp_content}")
            # -1, 1, 2
            model_choice = extract_choice_from_resp_pair_rm(resp_content)
        elif item["rm_tag"] == "4-way-pair_rm":
            question = make_prompt_4_way_pair_rm(
                item["question"], item["preds"][0], item["preds"][1], item["preds"][2], item["preds"][3]
            )
            question = question.replace("<image>\n", "")
            question = question.replace("<image>", "")
            if USE_COT == True:
                question += "\n\n" + FIXED_COT_PROMPT_DIRECT
            real_image_path = os.path.join(bench_img_root, item["images"][0])
            messages = message_format_with_image_file(question, real_image_path)
            resp = run_once_with_prompt_single_turn(
                model_client=client,
                model_name=MODEL_NAME,
                messages=messages,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            resp_content = resp.choices[0].message.content
            logger.info(f"resp_content:\n{resp_content}")
            # -1, 1, 2
            model_choice = extract_choice_from_resp_pair_rm(resp_content)
        else:
            raise ValueError(f"Unsupported rm_tag: {item['rm_tag']}")
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
    model_name_in_path = MODEL_NAME.split("/")[-1]
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
