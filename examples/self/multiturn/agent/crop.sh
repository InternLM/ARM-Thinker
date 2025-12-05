#!/bin/bash
# Exit immediately if a command exits with a non-zero status.
set -e
umask 000

# Activate Conda Environment
source /path/to/miniconda3/bin/activate verl
echo "[INFO] Conda env activated: $CONDA_DEFAULT_ENV"
# export PYTHONPATH=/path/to/fork/verl:$PYTHONPATH
# echo "[INFO] PYTHONPATH: $PYTHONPATH"


# Define Project Directories and Paths
PROJECT_DIR="/path/to/fork/verl"
CONFIG_PATH="$PROJECT_DIR/examples/self/multiturn/config"

# should before mkdir
cd $PROJECT_DIR

ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime

# Experiment Naming
PROJECT_NAME="ray"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
echo "[INFO] TIMESTAMP: $TIMESTAMP"
EXPERIMENT_NAME="crop--${TIMESTAMP}"

# Set Environment Variables for the Training Job
export VLLM_USE_V1=1
export MLFLOW_TRACKING_URI="sqlite:///$PROJECT_DIR/mlruns.db"
export VERL_LOGGING_LEVEL=INFO
export RAY_MASTER_PORT=6379
export RAY_DASHBOARD_PORT=8265

# Get Node Information from the Environment
export NODE_RANK=${NODE_RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"} # Default to localhost if not set
export RAY_ADDRESS="http://$MASTER_ADDR:$RAY_DASHBOARD_PORT"
export REWARD_MODEL_VERSION="v8"
export LC_REWARD_VERSION="v2"

use_kl_loss=False
kl_coef=0.0
use_kl_in_reward=False

echo "Using reward model version: $REWARD_MODEL_VERSION"
# ===================================================================
#                  MAIN LOGIC: HEAD vs WORKER
# ===================================================================


if [ "$NODE_RANK" -eq 0 ]; then
    ###################################
    # HEAD NODE LOGIC (NODE_RANK == 0)
    ###################################
    echo "[INFO] This is the HEAD node (Rank 0) with Master Address: $MASTER_ADDR"

    # Start Ray Head
    ray stop -f
    ray start --head --node-ip-address="$MASTER_ADDR" --port="$RAY_MASTER_PORT" --dashboard-host=0.0.0.0 --dashboard-port="$RAY_DASHBOARD_PORT" --num-gpus=8

    # Wait for all worker nodes to connect
    sleep 20

    LOG_FILE="logs/$PROJECT_NAME/$EXPERIMENT_NAME/log.txt"
    mkdir -p "$(dirname "$LOG_FILE")"
    
    # Submit the Ray job
        # --runtime-env-json="$RUNTIME_ENV_JSON" \
    ray job submit --address="$MASTER_ADDR:$RAY_MASTER_PORT" \
        -- python3 -m verl.trainer.main_ppo \
        --config-path="$CONFIG_PATH" \
        --config-name='multiturn_grpo_2_self_template' \
        data.train_files=$PROJECT_DIR/datasets/DeepEyes-Datasets-47k/tool_agent_all_pair_rm/train.parquet \
        data.val_files=[$PROJECT_DIR/datasets/DeepEyes-Datasets-47k/tool_agent_all_pair_rm/test.parquet] \
        data.return_multi_modal_inputs=False \
        actor_rollout_ref.rollout.multi_turn.tool_config_path="$PROJECT_DIR/examples/self/multiturn/config/tool_config/image_zoom_in_tool_config.yaml" \
        actor_rollout_ref.model.path=/path/to/sft-model \
        algorithm.adv_estimator=grpo \
        data.train_batch_size=512 \
        data.max_prompt_length=65000 \
        data.max_response_length=8192 \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        data.image_key=images \
        data.return_raw_chat=True \
        actor_rollout_ref.actor.optim.lr=2e-6 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=256 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
        actor_rollout_ref.actor.use_kl_loss=$use_kl_loss \
        actor_rollout_ref.actor.kl_loss_coef=$kl_coef \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.actor.checkpoint.save_contents=["hf_model"] \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
        actor_rollout_ref.rollout.n=16 \
        actor_rollout_ref.rollout.enforce_eager=True \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
        actor_rollout_ref.ref.fsdp_config.param_offload=False \
        actor_rollout_ref.rollout.mode=async \
        actor_rollout_ref.rollout.multi_turn.max_tool_response_length=4096 \
        actor_rollout_ref.rollout.multi_turn.max_assistant_turns=5 \
        actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
        algorithm.use_kl_in_reward=$use_kl_in_reward \
        trainer.critic_warmup=0 \
        trainer.logger='["console","tensorboard"]' \
        trainer.log_val_generations=5 \
        trainer.project_name="$PROJECT_NAME" \
        trainer.experiment_name="$EXPERIMENT_NAME" \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=2 \
        trainer.save_freq=8 \
        trainer.test_freq=8 \
        trainer.total_epochs=3 "$@" \
        2>&1 | tee >(sed -r "s/\x1B\[[0-9;]*[mK]//g" > "$LOG_FILE")
    
    echo "[HEAD] Job finished."
        # trainer.rollout_data_dir=$PROJECT_DIR/rollout_logs/$PROJECT_NAME/$EXPERIMENT_NAME/ \

else
    echo "[INFO] This is a WORKER node (Rank $NODE_RANK)."
    ray start --address="${MASTER_ADDR}:6379" --num-gpus=8

    echo "[WORKER] Started and connected to head node: ${MASTER_ADDR}:6379"
    echo "[WORKER] Monitoring head node status..."

    while true; do
        # if head node is not reachable, stop worker
        if ! ray status --address="${MASTER_ADDR}:6379" > /dev/null 2>&1; then
            echo "[WORKER] Head node unreachable. Stopping worker..."
            ray stop -f
            exit 0
        fi
        sleep 60  # check every 60 seconds
    done
fi
