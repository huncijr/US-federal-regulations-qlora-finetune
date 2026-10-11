"""Interactive terminal chat with a configured base model or saved LoRA adapter."""

import argparse
from pathlib import Path
from threading import Thread

from project_config import ROOT, latest_adapter, load_config, runtime_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=None, help="Override the configured local base checkpoint path.")
    parser.add_argument("--adapter", default=None, help="Saved adapter directory, or 'latest'. Omit for base-model chat.")
    parser.add_argument("--prompt", default=None, help="Generate one response and exit instead of opening an interactive chat.")
    parser.add_argument("--context", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    args = parser.parse_args()
    config = load_config(ROOT, args.config)
    paths = runtime_paths(config, ROOT)
    inference = config["inference"]
    args.context = args.context if args.context is not None else inference["context"]
    args.max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else inference["max_new_tokens"]
    if args.max_new_tokens < 1 or args.context <= args.max_new_tokens + 64:
        parser.error("Context must exceed max-new-tokens by more than 64 tokens.")

    model_dir = paths["model"]
    if args.model is not None:
        model_dir = args.model if args.model.is_absolute() else ROOT / args.model
    if not (model_dir / "config.json").is_file() or not list(model_dir.glob("*.safetensors")):
        parser.error("The configured local base checkpoint is missing. Update config.local.json or --model.")
    adapter_dir = None
    if args.adapter:
        adapter_dir = latest_adapter(paths["output"]) if args.adapter == "latest" else Path(args.adapter)
        if not adapter_dir.is_absolute():
            adapter_dir = ROOT / adapter_dir
        if not (adapter_dir / "adapter_config.json").is_file() or not (adapter_dir / "adapter_model.safetensors").is_file():
            parser.error("The adapter directory must contain adapter_config.json and adapter_model.safetensors.")
        metadata_file = adapter_dir.parent / "run_metadata.json"
        if metadata_file.is_file():
            import json
            metadata = json.loads(metadata_file.read_text())
            if config["model"]["id"] and metadata.get("model_id") != config["model"]["id"]:
                parser.error("Adapter and configured base-model identifiers differ. Select the training configuration.")

    import unsloth  # Import before Transformers to apply Unsloth patches.
    import torch
    from transformers import AutoTokenizer, TextIteratorStreamer
    from unsloth import FastLanguageModel

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable in this Python environment.")

    print("Loading local model. Stop the notebook kernel first to free GPU memory.")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_dir),
        max_seq_length=args.context,
        dtype=None,
        load_in_4bit=config["model"]["load_in_4bit"],
        full_finetuning=False,
        local_files_only=True,
        device_map={"": torch.cuda.current_device()},
    )
    if adapter_dir is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False, local_files_only=True)
        if (adapter_dir / "tokenizer_config.json").is_file():
            tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir), local_files_only=True)
        print("Loaded saved adapter:", adapter_dir.relative_to(ROOT) if adapter_dir.is_relative_to(ROOT) else adapter_dir)
    else:
        print("Using the base model without an adapter.")
    FastLanguageModel.for_inference(model)
    torch.manual_seed(inference["seed"])
    history = []
    budget = args.context - args.max_new_tokens
    print("Ready. Commands: /clear, /help, /exit. Ctrl+C also exits.")

    while True:
        try:
            question = args.prompt.strip() if args.prompt is not None else input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if not question:
            if args.prompt is not None:
                parser.error("The prompt must not be empty.")
            continue
        if args.prompt is None and question == "/exit":
            break
        if args.prompt is None and question == "/clear":
            history = []
            torch.manual_seed(inference["seed"])
            print("Conversation cleared.")
            continue
        if args.prompt is None and question == "/help":
            print("/clear: reset history; /exit: quit; /help: show commands.")
            continue

        messages = history + [{"role": "user", "content": question}]
        while True:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                **config["model"]["chat_template_kwargs"],
            )
            inputs = tokenizer(prompt, return_tensors="pt")
            if inputs["input_ids"].shape[1] <= budget:
                break
            if len(messages) > 1:
                del messages[:2]  # Remove the oldest complete conversation turn.
            else:
                break
        if inputs["input_ids"].shape[1] > budget:
            print("Message is too long. Shorten it or increase --context.")
            if args.prompt is not None:
                raise SystemExit(1)
            continue

        inputs = inputs.to(model.device)
        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=1,
        )
        errors = []

        def generate():
            try:
                with torch.inference_mode():
                    model.generate(
                        **inputs, streamer=streamer,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=True, temperature=inference["temperature"],
                        top_p=inference["top_p"], top_k=inference["top_k"],
                    )
            except Exception as exc:
                errors.append(exc)
                streamer.on_finalized_text("", stream_end=True)

        worker = Thread(target=generate, daemon=True)
        worker.start()
        print("Assistant: ", end="", flush=True)
        response = ""
        from queue import Empty

        try:
            while True:
                try:
                    text = next(streamer)
                except Empty:
                    if not worker.is_alive():
                        break
                    continue
                except StopIteration:
                    break
                response += text
                print(text, end="", flush=True)
            worker.join()
        except KeyboardInterrupt:
            print("\nExiting.")
            return
        print()
        if errors:
            print(f"Generation failed: {errors[0]}")
            if isinstance(errors[0], torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
            if args.prompt is not None:
                raise SystemExit(1)
            continue
        history = messages + [{"role": "assistant", "content": response}]
        if args.prompt is not None:
            break


if __name__ == "__main__":
    main()
