# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import datetime as dt

from ray._private.runtime_env.constants import RAY_JOB_CONFIG_JSON_ENV_VAR

# Generate unique timestamp for this training session
_timestamp = dt.datetime.now().strftime("%Y%m%d%H%M%S")
_tool_img_temp_dir = f"/tmp/verl_train_images_{_timestamp}"

PPO_RAY_RUNTIME_ENV = {
    "env_vars": {
        "TOKENIZERS_PARALLELISM": "true",
        "NCCL_DEBUG": "WARN",
        "VLLM_LOGGING_LEVEL": "WARN",
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "TOOL_CALL_IMG_TEMP": _tool_img_temp_dir,
    },
}


def get_ppo_ray_runtime_env():
    """
    A filter function to return the PPO Ray runtime environment.
    To avoid repeat of some environment variables that are already set.
    """
    working_dir = (
        json.loads(os.environ.get(RAY_JOB_CONFIG_JSON_ENV_VAR, "{}"))
        .get("runtime_env", {})
        .get("working_dir", None)
    )

    runtime_env = {
        "env_vars": PPO_RAY_RUNTIME_ENV["env_vars"].copy(),
        **({"working_dir": None} if working_dir is None else {}),
    }
    for key in list(runtime_env["env_vars"].keys()):
        if os.environ.get(key) is not None:
            runtime_env["env_vars"].pop(key, None)

    # Create the tool image temp directory
    tool_img_temp_dir = runtime_env["env_vars"].get("TOOL_CALL_IMG_TEMP")
    if tool_img_temp_dir:
        os.makedirs(tool_img_temp_dir, exist_ok=True)
        print(f"Created tool image temp directory: {tool_img_temp_dir}")

    return runtime_env
