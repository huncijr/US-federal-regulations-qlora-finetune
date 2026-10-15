"""Dataset validation and a one-call monitored QLoRA workflow."""

import json
from pathlib import Path


def load_examples(path):
    """Read JSONL messages; retain optional source metadata for provenance."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path.name}. Create curated messages-based JSONL data first. "
            "Raw eCFR baseline responses are not training targets."
        )
    examples = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            example = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name}:{number}: invalid JSON") from exc
        messages = example.get("messages") if isinstance(example, dict) else None
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"{path.name}:{number}: a nonempty messages list is required")
        expected = "user"
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                raise ValueError(f"{path.name}:{number}: each message must be an object")
            role, content = message.get("role"), message.get("content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError(f"{path.name}:{number}: message content must be nonempty text")
            if index == 0 and role == "system":
                continue
            if role != expected:
                raise ValueError(f"{path.name}:{number}: expected {expected}, received {role}")
            expected = "assistant" if expected == "user" else "user"
        if messages[-1]["role"] != "assistant":
            raise ValueError(f"{path.name}:{number}: the final message must be an assistant answer")
        examples.append(example)
    if not examples:
        raise ValueError(f"{path.name}: no examples found")
    return examples


def check_split_overlap(train, validation):
    """Reject exact duplicate conversations and repeated questions across splits.

    This is a basic check, not semantic near-duplicate detection. Document/topic
    separation and source verification still require dataset curation.
    """
    def signatures(examples):
        return [json.dumps(e["messages"], sort_keys=True, ensure_ascii=False)
                for e in examples]

    for name, examples in (("train", train), ("validation", validation)):
        values = signatures(examples)
        if len(set(values)) != len(values):
            raise ValueError(f"Duplicate conversations in the {name} split")
    if set(signatures(train)) & set(signatures(validation)):
        raise ValueError("Train/validation conversations overlap")
    def questions(examples):
        return {message["content"].strip() for example in examples
                for message in example["messages"] if message["role"] == "user"}

    if questions(train) & questions(validation):
        raise ValueError("Train/validation questions overlap")


def format_examples(examples, tokenizer, max_length, template_kwargs=None):
    """Use the native template and reject overlength samples instead of truncating."""
    texts = []
    for index, example in enumerate(examples, 1):
        formatted = tokenizer.apply_chat_template(
            example["messages"], tokenize=False,
            add_generation_prompt=False, **(template_kwargs or {}),
        )
        length = len(tokenizer(formatted, add_special_tokens=False)["input_ids"])
        if length > max_length:
            raise ValueError(
                f"Example {index} uses {length} tokens; limit is {max_length}. "
                "Split or shorten the example without losing its conditions and answer."
            )
        texts.append(formatted)
    return texts


def sample_validation(examples, limit, seed):
    """Choose a fixed evaluation sample; never sample from the final test set."""
    if not limit or len(examples) <= limit:
        return examples
    import random
    indices = sorted(random.Random(seed).sample(range(len(examples)), limit))
    return [examples[index] for index in indices]


def _dashboard_payload(url):
    from urllib.request import urlopen

    with urlopen(url + "/api/status", timeout=3) as response:
        payload = json.load(response)
    if not isinstance(payload.get("state"), dict):
        raise RuntimeError("The selected port is not serving the training dashboard.")
    return payload


def _ensure_dashboard(project, output_dir, port):
    """Reuse a local monitor, or start its lightweight server without loading a model."""
    import subprocess
    import sys
    import time
    from urllib.error import URLError

    url = f"http://127.0.0.1:{port}"
    try:
        _dashboard_payload(url)
        return url
    except URLError as exc:
        if not isinstance(exc.reason, ConnectionRefusedError):
            raise
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "dashboard.log").open("ab") as log:
        process = subprocess.Popen(
            [sys.executable, "-u", str(project / "monitoring.py"),
             "--port", str(port), "--log-dir", str(output_dir / "monitor")],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    for _ in range(30):
        if process.poll() is not None:
            raise RuntimeError("Dashboard exited. See dashboard.log under the output directory.")
        time.sleep(0.2)
        try:
            _dashboard_payload(url)
            return url
        except URLError:
            continue
    raise RuntimeError("Dashboard did not become available. Check its log before retrying.")


def run_finetuning(project=None, kernel_namespace=None, max_steps=None,
                   max_seq_length=None, lora_rank=None, seed=None,
                   dashboard_port=None, start_delay=5, config_path=None):
    """Run all preparation, monitored training, evaluation, and adapter saving.

    The local checkpoint must already be downloaded. No uploads occur. Each
    invocation saves to a unique run directory and releases its model afterward.
    Pass globals() from the notebook to detect models retained by manual cells.
    """
    import gc
    import hashlib
    import time
    from datetime import datetime, timezone
    from importlib.metadata import version
    from uuid import uuid4
    from monitoring import RunLog, create_training_callback
    from project_config import load_config, runtime_paths, validate_training_config

    wall_started = time.monotonic()
    project = Path(project or Path.cwd()).resolve()
    config = load_config(project, config_path)
    validate_training_config(config)
    paths = runtime_paths(config, project)
    training_config = config["training"]
    max_steps = max_steps if max_steps is not None else training_config["max_steps"]
    max_seq_length = max_seq_length if max_seq_length is not None else training_config["max_seq_length"]
    lora_rank = lora_rank if lora_rank is not None else training_config["lora_rank"]
    seed = seed if seed is not None else training_config["seed"]
    dashboard_port = dashboard_port if dashboard_port is not None else config["dashboard_port"]
    wall_limit = training_config.get("max_wall_seconds", 0)
    reserve = training_config.get("save_reserve_seconds", 180)
    if wall_limit < 0 or reserve < 0 or wall_limit and reserve >= wall_limit:
        raise ValueError("Time limit must be positive or disabled, with a smaller save reserve.")
    stop_deadline = wall_started + wall_limit - reserve if wall_limit else None
    eval_interval = training_config.get("eval_steps", 500)
    eval_limit = training_config.get("eval_max_samples", 512)
    if eval_interval < 1 or eval_limit < 1:
        raise ValueError("Evaluation interval and sample size must be positive.")
    if wall_limit and training_config.get("final_eval_full", False):
        raise ValueError("Disable final_eval_full for a bounded-time run; evaluate the full set separately.")
    namespace = kernel_namespace if kernel_namespace is not None else {}
    if namespace.get("_finetuning_running"):
        raise RuntimeError("A fine-tuning invocation is already running in this kernel.")
    if namespace.get("model") is not None or namespace.get("trainer") is not None:
        raise RuntimeError("A manual model/Trainer remains in memory. Save it if needed, then Restart Kernel before using this launcher.")
    if not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    if max_seq_length < 1 or lora_rank < 1 or start_delay < 0:
        raise ValueError("Invalid sequence length, LoRA rank, or start delay")
    if not (project / "monitoring.py").is_file():
        raise RuntimeError("Run from the repository root containing monitoring.py.")

    model_id = config["model"]["id"]
    model_dir = paths["model"]
    if not list(model_dir.glob("*.safetensors")):
        raise FileNotFoundError("Download the checkpoint into the configured model path first.")
    files = {name: paths[name] for name in ("train", "validation", "test")}
    output_dir = paths["output"]
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8]
    run_dir = output_dir / "training-runs" / run_id
    callback = stage_log = model = tokenizer = trainer = torch = None
    namespace["_finetuning_running"] = True
    try:
        print("[1/6] Checking configuration and connecting the live dashboard.", flush=True)
        dashboard_url = _ensure_dashboard(project, output_dir, dashboard_port)
        print(f"Open this page now: {dashboard_url}", flush=True)
        stage_log = RunLog(output_dir / "monitor")
        stage_log.update("setup", status="checking_datasets", total_steps=max_steps)

        print("[2/6] Loading and validating all three datasets.", flush=True)
        examples = {name: load_examples(path) for name, path in files.items()}
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
            check_split_overlap(examples[left], examples[right])
        print("Dataset sizes:", {name: len(rows) for name, rows in examples.items()}, flush=True)
        evaluation_examples = sample_validation(examples["validation"], eval_limit, seed)
        print(f"Periodic/final validation sample: {len(evaluation_examples)} examples", flush=True)

        print("[3/6] Loading the configured local model once.", flush=True)
        stage_log.update("loading", status="loading_model")
        import unsloth  # Apply patches before importing the training libraries.
        import torch
        from unsloth import FastLanguageModel
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in the selected Python environment.")
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        print(f"Free VRAM: {free / 1024**3:.2f} / {total / 1024**3:.2f} GiB", flush=True)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(model_dir), max_seq_length=max_seq_length,
            dtype=None, load_in_4bit=config["model"]["load_in_4bit"], full_finetuning=False,
            local_files_only=True, device_map={"": torch.cuda.current_device()},
        )

        print("[4/6] Preparing the adapter, assistant-only loss, and monitor.", flush=True)
        stage_log.update("preparing", status="preparing_trainer")
        from datasets import Dataset
        from trl import SFTConfig, SFTTrainer
        from unsloth.chat_templates import train_on_responses_only
        template_kwargs = config["model"]["chat_template_kwargs"]
        train_text = format_examples(examples["train"], tokenizer, max_seq_length, template_kwargs)
        validation_text = format_examples(evaluation_examples, tokenizer, max_seq_length, template_kwargs)
        model = FastLanguageModel.get_peft_model(
            model, r=lora_rank, lora_alpha=2 * lora_rank, lora_dropout=0,
            target_modules=training_config["lora_target_modules"],
            bias="none", use_gradient_checkpointing="unsloth", random_state=seed,
        )
        trainer = SFTTrainer(
            model=model, processing_class=tokenizer,
            train_dataset=Dataset.from_dict({"text": train_text}),
            eval_dataset=Dataset.from_dict({"text": validation_text}),
            args=SFTConfig(
                output_dir=str(run_dir / "checkpoints"), dataset_text_field="text",
                max_length=max_seq_length, packing=False, dataset_num_proc=1,
                per_device_train_batch_size=1, per_device_eval_batch_size=1,
                gradient_accumulation_steps=training_config["gradient_accumulation_steps"], max_steps=max_steps,
                learning_rate=training_config["learning_rate"], warmup_steps=max(1, max_steps // 10), optim="adamw_8bit",
                bf16=torch.cuda.is_bf16_supported(), fp16=not torch.cuda.is_bf16_supported(),
                logging_steps=1, logging_nan_inf_filter=False,
                include_num_input_tokens_seen="non_padding",
                eval_strategy="steps", eval_steps=min(max_steps, eval_interval),
                save_strategy="no", report_to="none", seed=seed,
            ),
        )
        trainer = train_on_responses_only(
            trainer, instruction_part=training_config["instruction_marker"],
            response_part=training_config["response_marker"],
        )
        for split in (trainer.train_dataset, trainer.eval_dataset):
            if any(not any(label != -100 for label in row["labels"]) for row in split):
                raise ValueError("An example has no assistant tokens contributing to loss.")
        callback = create_training_callback(output_dir / "monitor", len(examples["train"]),
                                            model, max_steps=max_steps, run_log=stage_log,
                                            stop_deadline=stop_deadline,
                                            run_started_at=wall_started,
                                            max_wall_seconds=wall_limit)
        callback.log.update("budget", validation_examples=len(evaluation_examples),
                            max_wall_seconds=wall_limit, save_reserve_seconds=reserve)
        trainer.add_callback(callback)
        payload = _dashboard_payload(dashboard_url)
        if payload["state"].get("run_id") != callback.log.state["run_id"]:
            raise RuntimeError("Dashboard is following another directory. Restart it with --log-dir pointing to the configured output directory's monitor folder.")
        print(f"Dashboard ready at 0/{max_steps}. Training starts in {start_delay} seconds.", flush=True)
        time.sleep(start_delay)
        if stop_deadline is not None and time.monotonic() >= stop_deadline:
            raise RuntimeError("Setup exhausted the training time budget. No optimizer updates were started.")

        print(f"[5/6] Training for {max_steps} optimizer updates, then validating.", flush=True)
        torch.cuda.reset_peak_memory_stats()
        training = trainer.train()
        validation = next((entry for entry in reversed(trainer.state.log_history)
                           if "eval_loss" in entry and entry.get("step") == trainer.state.global_step), None)
        if validation is None:
            if wall_limit and time.monotonic() >= wall_started + wall_limit - 120:
                validation = next((entry for entry in reversed(trainer.state.log_history) if "eval_loss" in entry),
                                  {"evaluation_skipped": True, "reason": "save_time_reserved"})
            else:
                validation = trainer.evaluate()
        if training_config.get("final_eval_full", False) and len(evaluation_examples) != len(examples["validation"]):
            full_text = format_examples(examples["validation"], tokenizer, max_seq_length, template_kwargs)
            validation = trainer.evaluate(eval_dataset=Dataset.from_dict({"text": full_text}))
        callback.set_status("evaluated")

        print("[6/6] Saving the adapter, metrics, and provenance.", flush=True)
        run_dir.mkdir(parents=True, exist_ok=True)
        adapter_dir = run_dir / "adapter"
        model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        for filename, metrics in (("train_metrics.json", training.metrics), ("validation_metrics.json", validation)):
            (run_dir / filename).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        revision_file = paths["revision"]
        metadata = {
            "model_id": model_id, "checkpoint_provenance": revision_file.read_text() if revision_file.exists() else "not recorded",
            "config": config,
            "max_steps": max_steps, "max_seq_length": max_seq_length, "lora_rank": lora_rank, "seed": seed,
            "actual_optimizer_steps": trainer.state.global_step,
            "validation_sample_count": len(evaluation_examples),
            "validation_sample_question_hash": hashlib.sha256(json.dumps([row["messages"][0]["content"] for row in evaluation_examples]).encode()).hexdigest(),
            "max_wall_seconds": wall_limit, "whole_run_seconds_before_save": time.monotonic() - wall_started,
            "stop_reason": callback.log.state.get("stop_reason", "planned_steps_completed"),
            "dataset_counts": {name: len(rows) for name, rows in examples.items()},
            "dataset_hashes": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()},
            "packages": {name: version(name) for name in ("torch", "unsloth", "transformers", "bitsandbytes", "peft", "trl", "datasets")},
            "monitor_run": callback.log.directory.name, "final_test_evaluated": False,
        }
        (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        relative_adapter = str(adapter_dir.relative_to(project)) if adapter_dir.is_relative_to(project) else str(adapter_dir)
        callback.set_status("saved", adapter_path=relative_adapter)
        print("Completed. Adapter saved:", relative_adapter, flush=True)
        print("Final-test factual accuracy has not been evaluated.", flush=True)
        # Subsequent manual notebook cells must not start a second training run.
        namespace.update(LOAD_MODEL=False, RUN_TRAINING=False, DOWNLOAD_MODEL=False)
        return {"adapter_path": relative_adapter, "train_metrics": training.metrics,
                "validation_metrics": validation, "dashboard_url": dashboard_url}
    except BaseException as exc:
        logger = callback.log if callback is not None else stage_log
        if logger is not None:
            logger.update("failure", status="interrupted" if isinstance(exc, KeyboardInterrupt) else "error", error=str(exc))
        raise
    finally:
        trainer = callback = tokenizer = model = None
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        namespace["_finetuning_running"] = False
