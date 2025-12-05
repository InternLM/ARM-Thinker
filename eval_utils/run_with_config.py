import argparse
import os
import subprocess
import yaml
import sys
from dotenv import load_dotenv

# Keys to exclude from CLI arguments
IGNORED_KEYS = {"script"}

load_dotenv()
PROJECT_ROOT = os.getenv("PROJECT_ROOT")
if PROJECT_ROOT is None:
    print("[ERROR] PROJECT_ROOT is not set")
    exit(1)
print(f"Load project root env: {PROJECT_ROOT}")


def run_single_config(run_name, config):
    run_config = config.get("runs", {}).get(run_name)
    if not run_config:
        print(f"[ERROR] Config '{run_name}' not found in config.yaml")
        return False

    script = run_config.get("script")
    if not script:
        print(f"[ERROR] Missing 'script' field in config for '{run_name}'")
        return False

    # Build env
    env = os.environ.copy()
    env["PYTHONPATH"] = PROJECT_ROOT

    # Build command
    cmd_parts = [f"python {script}"]
    for key, value in run_config.items():
        if key in IGNORED_KEYS:
            continue
        if isinstance(value, bool):
            if value:
                cmd_parts.append(f"--{key}")
        else:
            cmd_parts.append(f"--{key} {value}")

    # Add run_name
    cmd_parts.append(f"--run_name {run_name}")

    # Add log file
    model_name = run_config.get("model", "default_model").replace("/", "_")
    log_file = f"logs/{run_name}/{model_name}.log"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    cmd_str = " ".join(cmd_parts) + f" 2>&1 | tee -a {log_file}"

    print("=" * 80)
    print("Running:", run_name)
    print("Shell command:", cmd_str)
    print("=" * 80)

    # exit()

    # Run command
    result = subprocess.run(cmd_str, shell=True, env=env)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", nargs="+", help="Name(s) of the run config(s)")
    args = parser.parse_args()

    # Load config.yaml
    with open("eval_utils/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Loop over run_names
    for run_name in args.run_name:
        success = run_single_config(run_name, config)
        if success:
            print(f"[OK] Finished {run_name}")
        else:
            print(f"[FAIL] {run_name} (but continuing...)")


if __name__ == "__main__":
    main()
