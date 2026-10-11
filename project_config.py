"""Load model-independent defaults and ignored local experiment settings."""

import json
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _merge(base, changes):
    result = deepcopy(base)
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(project=None, config_path=None):
    project = Path(project or ROOT).resolve()
    example = project / "config.example.json"
    if not example.is_file():
        raise FileNotFoundError("Missing config.example.json in the project root.")
    config = json.loads(example.read_text(encoding="utf-8"))
    local = Path(config_path) if config_path else project / "config.local.json"
    if not local.is_absolute():
        local = project / local
    if config_path and not local.is_file():
        raise FileNotFoundError(f"Missing configuration: {local.name}")
    if local.is_file():
        config = _merge(config, json.loads(local.read_text(encoding="utf-8")))
    return config


def runtime_paths(config, project=None):
    project = Path(project or ROOT).resolve()

    def resolve(value):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Configured paths must be nonempty strings.")
        path = Path(value).expanduser()
        return path if path.is_absolute() else project / path

    return {"model": resolve(config["model"]["path"]),
            "revision": resolve(config["model"]["revision_file"]),
            "train": resolve(config["data"]["train"]),
            "validation": resolve(config["data"]["validation"]),
            "test": resolve(config["data"]["test"]),
            "output": resolve(config["output_dir"])}


def validate_training_config(config):
    training = config["training"]
    if not training.get("instruction_marker") or not training.get("response_marker"):
        raise ValueError("Configure instruction_marker and response_marker for the selected tokenizer before training.")
    if training["instruction_marker"] == training["response_marker"]:
        raise ValueError("User and assistant markers must be distinct.")
    targets = training.get("lora_target_modules")
    if not isinstance(targets, (str, list)) or not targets:
        raise ValueError("Configure supported LoRA target modules for the selected model.")


def latest_adapter(output_dir):
    """Select the newest completed run with both configuration and adapter weights."""
    output_dir = Path(output_dir)
    candidates = [path for path in output_dir.glob("training-runs/*/adapter")
                  if (path / "adapter_config.json").is_file()
                  and (path / "adapter_model.safetensors").is_file()
                  and (path.parent / "run_metadata.json").is_file()]
    if candidates:
        return max(candidates, key=lambda path: path.parent.name)
    legacy = output_dir / "adapter"
    if (legacy / "adapter_config.json").is_file() and (legacy / "adapter_model.safetensors").is_file():
        return legacy
    raise FileNotFoundError("No saved adapter found in the configured output directory.")
