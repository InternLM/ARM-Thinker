from verl.utils.import_utils import deprecated


def default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
):
    """Compute the score for a given solution based on the data source.

    Args:
        data_source (str): The source dataset identifier which determines the scoring method.
        solution_str (str): The solution string to be evaluated.
        ground_truth (str): The ground truth answer for comparison.
        extra_info (dict, optional): Additional information that might be needed for scoring. Defaults to None.

    Returns:
        float: The computed score as a floating point number. If the result is a dictionary,
               it returns the dictionary instead.

    Raises:
        NotImplementedError: If the reward function is not implemented for the given data source.
    """
    if data_source == "openai/gsm8k" or "openai/gsm8k" in data_source:
        from . import gsm8k

        res = gsm8k.compute_score(solution_str, ground_truth)
    elif data_source in [
        "lighteval/MATH",
        "DigitalLearningGmbH/MATH-lighteval",
        "HuggingFaceH4/MATH-500",
    ]:
        from . import math

        res = math.compute_score(solution_str, ground_truth)
        # [Optional] Math-Verify Integration
        # For enhanced accuracy, consider utilizing Math-Verify (https://github.com/huggingface/Math-Verify).
        # Note: Math-Verify needs to be manually installed via pip: `pip install math-verify`.
        # To use it, override the `compute_score` function with the following implementation:

        # from . import math_verify
        # res = math_verify.compute_score(solution_str, ground_truth)
    elif data_source == "math_dapo" or data_source.startswith("aime"):
        from . import math_dapo

        res = math_dapo.compute_score(solution_str, ground_truth)
    elif data_source in [
        "numina_aops_forum",
        "numina_synthetic_math",
        "numina_amc_aime",
        "numina_synthetic_amc",
        "numina_cn_k12",
        "numina_olympiads",
    ]:
        from . import prime_math

        res = prime_math.compute_score(solution_str, ground_truth)
    elif data_source in ["codecontests", "apps", "codeforces", "taco"]:
        # Use the passed sandbox_fusion_url if available
        if sandbox_fusion_url:
            from . import sandbox_fusion

            # Pass the URL directly, ground_truth likely contains test cases here
            res = sandbox_fusion.compute_score(
                sandbox_fusion_url,
                concurrent_semaphore,
                memory_limit_mb,
                solution_str,
                ground_truth,
                continuous=True,
            )
        else:
            # If no sandbox URL is provided, fall back to prime_code or raise error
            from . import prime_code

            # Assuming prime_code doesn't need the URL
            res = prime_code.compute_score(solution_str, ground_truth, continuous=True)
    elif data_source in ["hiyouga/geometry3k"] or "hiyouga/geometry3k" in data_source:
        from . import geo3k

        res = geo3k.compute_score(solution_str, ground_truth)
    elif data_source in [
        "searchR1_nq",
        "searchR1_triviaqa",
        "searchR1_popqa",
        "searchR1_hotpotqa",
        "searchR1_2wikimultihopqa",
        "searchR1_musique",
        "searchR1_bamboogle",
    ]:
        from . import search_r1_like_qa_em

        res = search_r1_like_qa_em.compute_score(solution_str, ground_truth)
    elif data_source in ["mmif", "mmif_instruction_following"] or "mmif" in data_source:
        from . import mmif

        res = mmif.compute_score(solution_str, ground_truth, extra_info)
    elif data_source == "llava_critic":
        import os

        rm_version = os.getenv("LC_REWARD_VERSION", "v3")
        if rm_version == "v2":
            from . import llava_critic_reward_v2 as llava_critic_reward
        elif rm_version == "v3":
            from . import llava_critic_reward_v3 as llava_critic_reward
        elif rm_version == "v4":
            from . import pair_rm_with_tool_reward_v8 as llava_critic_reward
        else:
            from . import llava_critic_reward as llava_critic_reward

        res = llava_critic_reward.compute_score(solution_str, ground_truth, extra_info)
    elif data_source == "mp_docvqa":
        import os

        rm_version = os.getenv("MP_DOCVQA_REWARD_VERSION", "v1")
        if rm_version == "v1":
            from . import mp_docvqa_reward_v1 as mp_docvqa_reward
        else:
            raise ValueError(
                f"Unsupported MP_DOCVQA_REWARD_VERSION: {rm_version}. Supported versions: v1"
            )
        res = mp_docvqa_reward.compute_score(solution_str, ground_truth, extra_info)
    elif data_source == "val_deepeyes_crop":
        from . import val_deepeyes_crop_reward
        res = val_deepeyes_crop_reward.compute_score(solution_str, ground_truth, extra_info)
    elif data_source == "val_llava_critic":
        from . import val_llava_critic_reward
        res = val_llava_critic_reward.compute_score(solution_str, ground_truth, extra_info)
    elif data_source == "deepeyes_crop":
        import os

        # Get reward model version from environment variable, default to v4
        rm_version = os.getenv("REWARD_MODEL_VERSION", "v8")

        if rm_version == "v8":
            from . import pair_rm_with_tool_reward_v8 as pair_rm

            print(f"Using reward model version: {rm_version}")
        # elif rm_version == "v9":
        #     from . import pair_rm_with_tool_reward_v9_only_acc as pair_rm

        #     print(f"Using reward model version: {rm_version}")
        # elif rm_version == "v10":
        #     from . import pair_rm_with_tool_reward_v10_stable_tool_reward as pair_rm

            print(f"Using reward model version: {rm_version}")
        else:
            raise ValueError(
                f"Unsupported REWARD_MODEL_VERSION: {rm_version}. Supported versions: v1, v2, v3, v4, v5, v6, v7, v8, v9, v10"
            )

        res = pair_rm.compute_score(solution_str, ground_truth, extra_info)

    else:
        raise NotImplementedError(
            f"Reward function is not implemented for {data_source=}"
        )

    if isinstance(res, dict):
        return res
    elif isinstance(res, int | float | bool):
        return float(res)
    else:
        return float(res[0])


@deprecated("verl.utils.reward_score.default_compute_score")
def _default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
):
    """
    Legacy function API to be deprecated. Please use `default_compute_score` instead.
    """
    return default_compute_score(
        data_source,
        solution_str,
        ground_truth,
        extra_info,
        sandbox_fusion_url,
        concurrent_semaphore,
        memory_limit_mb,
    )


__all__ = ["default_compute_score"]
