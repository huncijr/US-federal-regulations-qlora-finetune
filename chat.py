"""Interactive terminal chat with the local Qwen3 4B checkpoint."""

import argparse
from pathlib import Path
from threading import Thread


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()
    if args.max_new_tokens < 1 or args.context <= args.max_new_tokens + 64:
        parser.error("Context must exceed max-new-tokens by more than 64 tokens.")

    import unsloth  # Import before Transformers to apply Unsloth patches.
    import torch
    from transformers import TextIteratorStreamer
    from unsloth import FastLanguageModel

    model_dir = Path(__file__).resolve().parent / "models" / "qwen3-4b"
    if not (model_dir / "model.safetensors").exists():
        raise SystemExit(f"Local checkpoint is missing: {model_dir}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable in this Python environment.")

    print("Loading local model. Stop the notebook kernel first to free GPU memory.")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_dir),
        max_seq_length=args.context,
        dtype=None,
        load_in_4bit=True,
        full_finetuning=False,
        local_files_only=True,
    )
    FastLanguageModel.for_inference(model)
    history = []
    budget = args.context - args.max_new_tokens
    print("Ready. Commands: /clear, /help, /exit. Ctrl+C also exits.")

    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if not question:
            continue
        if question == "/exit":
            break
        if question == "/clear":
            history = []
            print("Conversation cleared.")
            continue
        if question == "/help":
            print("/clear: reset history; /exit: quit; /help: show commands.")
            continue

        messages = history + [{"role": "user", "content": question}]
        while True:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
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
                        do_sample=True, temperature=0.7, top_p=0.8, top_k=20,
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
            continue
        history = messages + [{"role": "assistant", "content": response}]


if __name__ == "__main__":
    main()
