import os
import random
import re
import csv
import json
import httpx
import logging
import traceback
import argparse
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import Dataset
from openai import OpenAI
from eval_utils.reward_bench_2.utils import process_single_model

from eval_utils.reward_bench_2.utils import format_4_way_judge_answers, extract_winner
from eval_utils.utils import BENCH_LIST, FIXED_COT_PROMPT_DIRECT

COT_PT_TAG = os.environ.get("COT_PT_TAG", "0")

# ==================== CONFIG ====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--workers", type=int, default=8)
parser.add_argument("--base", type=str, default="xxx")
parser.add_argument("--key", type=str, default="sk-xxx")
parser.add_argument("--model", type=str, default="gpt-4.1")
parser.add_argument("--proxy", type=bool, default=False)
parser.add_argument("--bench_split", type=str, default="")
parser.add_argument("--work_dir", type=str, default="./eval_results")
parser.add_argument("--score_w_ratings", type=bool, default=False)
args = parser.parse_args()

if not args.proxy:
    os.environ.update(
        {"http_proxy": "", "https_proxy": "", "HTTP_PROXY": "", "HTTPS_PROXY": ""}
    )

client = OpenAI(api_key=args.key, base_url=args.base, http_client=httpx.Client(verify=False))
model_name = args.model


# ==================== PROMPT TEMPLATES ====================
ratings_prompt_ties = """\
### Task Description
Please act as an impartial judge and evaluate the quality of the response provided by an
AI assistant to the user query displayed below.

Notes:
1- Your evaluation should consider factors such as the helpfulness, relevance, and accuracy of the response, but need not consider depth or level of detail of the response.
2- Begin your evaluation by providing a short explanation.
3- Be as objective as possible. After providing your explanation, please rate the response on a scale of 1 to 10. For your rating, only give a number between 1 and 10 (inclusive), do not use any markdown, and do not put any text after your final rating.

[Query]
{prompt}

[Response]
{completion}

[Your judgement]"""


# ==================== HELPERS ====================
def message_format(system_prompt, user_prompt):
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
    ]


def stable_hash_to_seed(s):
    import hashlib
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16) % (2**32)


def run_once_with_prompt_single_turn(system_prompt, user_prompt, retry=3):
    messages = message_format(system_prompt, user_prompt)
    for _ in range(retry):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=7000,
                extra_body={"repetition_penalty": 1.2},
            )
            return resp
        except Exception as e:
            logger.warning(f"Retrying after error: {e}")
    raise RuntimeError("Failed after retries")


# ==================== CORE LOGIC ====================
def process_item(item):
    """
    - Non-Ties: 4-way pairwise judgment;
    - Ties: rating score (only save ratings, no aggregation).
    """
    judge_result = {}
    subset = item["subset"]

    try:
        # ---------- Non-Ties ----------
        if not args.score_w_ratings and subset != "Ties":
            prompt = item["prompt"]
            chosen = item["chosen"]
            rejected = item["rejected"]

            a, b, c, d = chosen[0], rejected[0], rejected[1], rejected[2]
            item_id = item.get("id", 0)
            rng = np.random.default_rng(seed=stable_hash_to_seed(str(item_id)))
            shuffle_option = rng.integers(0, 4)

            if shuffle_option == 0:
                winner_text, losers = "A", ["B", "C", "D"]
            elif shuffle_option == 1:
                a, b = b, a
                winner_text, losers = "B", ["A", "C", "D"]
            elif shuffle_option == 2:
                a, c = c, a
                winner_text, losers = "C", ["A", "B", "D"]
            else:
                a, d = d, a
                winner_text, losers = "D", ["A", "B", "C"]

            system_prompt, user_prompt = format_4_way_judge_answers(
                prompt, a, b, c, d, args.model
            )
            if COT_PT_TAG == "1":
                user_prompt += "\n\n" + FIXED_COT_PROMPT_DIRECT
            resp = run_once_with_prompt_single_turn(system_prompt, user_prompt)
            msg = resp.model_dump()["choices"][0]["message"]
            logger.info(f"resp:\n{msg}")
            winner_resp = extract_winner(msg["content"])
            logger.info(f"winner_resp:\n{winner_resp}")
            score = 1 if winner_resp == winner_text else 0 if winner_resp in losers else 0.25
            item["score"] = score

            judge_result.update({
                "model_resp": msg,
                "model_judgement": winner_resp,
                "label": winner_text,
                "score": score,
            })
            return judge_result

        # ---------- Ties ----------
        else:
            prompt = item["prompt"]
            all_answers = item["chosen"] + item["rejected"]
            ratings, raw = [], []

            for ans in all_answers:
                user_prompt = ratings_prompt_ties.format(prompt=prompt, completion=ans)
                if COT_PT_TAG == "1":
                    user_prompt += "\n\n" + FIXED_COT_PROMPT_DIRECT
                system_prompt = ""
                try:
                    resp = run_once_with_prompt_single_turn(system_prompt, user_prompt)
                    content = resp.model_dump()["choices"][0]["message"]["content"].strip()
                    # m = re.search(r"\b([1-9]|10)\b\s*$", content)
                    # rating = int(m.group(1)) if m else -1

                    # [TAG-Chris]
                    # clean <think> and <answer>... tags
                    cleaned = re.sub(r"<[^>]+>", "", content).strip()
                    numbers = re.findall(r"\b([1-9]|10)\b", cleaned)
                    rating = int(numbers[-1]) if numbers else -1
                except Exception as e:
                    logger.error(f"Ties rating error: {e}")
                    rating, content = -1, "$ERROR$"

                ratings.append(rating)
                raw.append(content)

            item["tie_ratings"] = ratings
            item["tie_judgements"] = raw
            judge_result.update({"ratings": ratings, "raw_judgements": raw})
            return judge_result

    except Exception as e:
        logger.error(f"process_item error: {e}")
        logger.error(traceback.format_exc())
        raise


# ==================== MAIN ====================
if __name__ == "__main__":
    split_name = args.bench_split or list(BENCH_LIST.keys())[0]
    bench_path = BENCH_LIST[split_name]["path"]
    model_tag = model_name.split("/")[-1]

    # ---- read bench data ----
    with open(bench_path, "r") as f:
        bench_data = [json.loads(line) for line in f]

    save_dir = f"{args.work_dir}/{split_name}/{model_tag}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/{model_tag}.jsonl"

    # ---- skip done items ----
    results = []
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            done = [json.loads(line) for line in f]
        done_ids = {x["id"] for x in done}
        bench_data = [x for x in bench_data if x["id"] not in done_ids]
        results.extend(done)

    logger.info(f"Left {len(bench_data)} items to process")

    # ---- multi-thread execution ----
    import threading
    lock = threading.Lock()

    # debug = True
    # if debug:
    #     bench_data = bench_data[:10]

    with ThreadPoolExecutor(max_workers=args.workers) as pool, open(save_path, "a", encoding="utf-8") as f:
        futures = {pool.submit(process_item, x): x for x in bench_data}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
            try:
                res = fut.result()
                item = futures[fut]
                item["judge_result"] = res
                results.append(item)
                # thread-safe write
                with lock:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    f.flush()
            except Exception as e:
                logger.error(f"Error in future: {e}")
                os._exit(1)

    # ==================== SCORING ====================
    rows = [["Model", "Dataset", "Score", "Total Items"]]
    subsets = sorted({x["subset"] for x in results})

    # ---- process Ties subset ----
    ties_rows = []
    for x in results:
        if x["subset"] == "Ties" and "judge_result" in x and "ratings" in x["judge_result"]:
            ties_rows.append({
                "id": x["id"],
                "scores": x["judge_result"]["ratings"],
                "num_correct": x.get("num_correct", 1),
            })

    ties_score = None
    if ties_rows:
        ds = Dataset.from_list(ties_rows)
        _, ties_score = process_single_model(ds)

    # ---- process each subset ----
    subset_scores = {}
    for subset in subsets:
        if subset == "Ties" and ties_score is not None:
            subset_scores[subset] = ties_score
            rows.append([model_tag, f"{split_name}/{subset}", ties_score, len(ties_rows)])
            logger.info(f"{subset} official score: {ties_score:.4f}")
        else:
            s = [x.get("score", 0) for x in results if x.get("subset") == subset and "score" in x]
            avg = sum(s) / len(s) if s else 0.0
            subset_scores[subset] = avg
            rows.append([model_tag, f"{split_name}/{subset}", avg, len(s)])
            logger.info(f"{subset} avg: {avg:.4f}")

    # ---- calculate sample-level averages ----
    all_scores = []
    non_ties_scores = []
    for item in results:
        if item.get("subset") == "Ties":
            if ties_score is not None:
                all_scores.append(ties_score)
        else:
            if "score" in item:
                all_scores.append(item["score"])
                non_ties_scores.append(item["score"])

    sample_avg_incl_ties = sum(all_scores) / len(all_scores) if all_scores else 0.0
    sample_avg_excl_ties = sum(non_ties_scores) / len(non_ties_scores) if non_ties_scores else 0.0

    # ---- calculate subset-level averages ----
    subset_avg_incl_ties = sum(subset_scores.values()) / len(subset_scores) if subset_scores else 0.0
    subset_avg_excl_ties = sum(
        [v for k, v in subset_scores.items() if k != "Ties"]
    ) / (len(subset_scores) - 1) if len(subset_scores) > 1 else 0.0

    # ---- logging ----
    logger.info(f"==== Sample-level average (incl. Ties): {sample_avg_incl_ties:.4f} ====")
    logger.info(f"==== Sample-level average (excl. Ties): {sample_avg_excl_ties:.4f} ====")
    logger.info(f"==== Subset-level average (incl. Ties): {subset_avg_incl_ties:.4f} ====")
    logger.info(f"==== Subset-level average (excl. Ties): {subset_avg_excl_ties:.4f} ====")

    # ---- write summary rows ----
    rows.append(["", "SAMPLE_AVG_INCL_TIES", sample_avg_incl_ties, len(all_scores)])
    rows.append(["", "SAMPLE_AVG_EXCL_TIES", sample_avg_excl_ties, len(non_ties_scores)])
    rows.append(["", "SUBSET_AVG_INCL_TIES", subset_avg_incl_ties, len(subset_scores)])
    rows.append(["", "SUBSET_AVG_EXCL_TIES", subset_avg_excl_ties, len(subset_scores) - 1])

    # ---- write to CSV ----
    csv_path = f"{save_dir}/{model_tag}_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)

    logger.info(f"Summary saved to {csv_path}")

    # randomly select an item with subset not Ties
    temp_item = random.choice([x for x in results if x["subset"] != "Ties"])
    system_prompt, user_prompt = format_4_way_judge_answers(
        temp_item["prompt"],
        temp_item["chosen"][0],
        temp_item["rejected"][0],
        temp_item["rejected"][1],
        temp_item["rejected"][2],
        args.model,
    )
    if COT_PT_TAG == "1":
        user_prompt += "\n\n" + FIXED_COT_PROMPT_DIRECT
    messages = message_format(system_prompt, user_prompt)
    with open(
        f"{args.work_dir}/{split_name}/{model_tag}/prompt.json",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(json.dumps(messages, ensure_ascii=False))

    logger.info(f"Prompt saved to {f.name}")


