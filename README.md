# Configurable QLoRA workflow

A local, configuration-driven workflow for supported text language models:
conversational dataset validation, QLoRA training, live monitoring, adapter saving,
and terminal inference. Both a one-call launcher and manual notebook sections are available.

## Configuration

For a new checkout, create `config.local.json` using `config.example.json` as a
starting point. Set the local checkpoint path, optional upstream identifier,
dataset paths, output directory, tokenizer template options, LoRA target modules,
and user/assistant loss-mask markers for your selected model.

The local configuration is ignored by Git. Public code contains no fixed model
identity or personal filesystem path. Architecture support still depends on
Unsloth/Transformers/PEFT; a configurable workflow is not universal model support.

Use a compatible environment with CUDA-enabled PyTorch, Unsloth, Transformers,
BitsAndBytes, PEFT, TRL, Datasets, Hugging Face Hub, and Jupyter support.

## Training and monitoring

Open `finetune.ipynb` at the repository root and select the project environment.
After restarting the kernel, run the **Run all steps** cell for automatic
preparation, monitoring, training, validation, and saving. Alternatively, skip
the shortcut and run the numbered manual sections.

The default ten optimizer steps are a smoke test, not ten epochs. Source verification
and question-answer review are separate from schema validation. Keep source groups
and near-duplicate tasks within the same split, and do not tune on the final test.

The shortcut starts or reuses the local dashboard. To run it yourself:

```bash
.venv/bin/python monitoring.py
```

The dashboard reads the configured output directory and displays training status,
loss, gradient norm, learning rate, epochs, updates, memory, tokens, and timing.
Factual task accuracy remains unevaluated until a separately reviewed test report exists.

## Testing a saved adapter

Base-model chat:

```bash
.venv/bin/python chat.py
```

Latest completed adapter from the configured output directory:

```bash
.venv/bin/python chat.py --adapter latest
```

Specific adapter:

```bash
.venv/bin/python chat.py --adapter outputs/training/training-runs/RUN_ID/adapter
```

A single response for a controlled comparison:

```bash
.venv/bin/python chat.py --prompt "Your evaluation question"
.venv/bin/python chat.py --adapter latest --prompt "Your evaluation question"
```

Use identical configuration, prompt, seed, and response limits for comparisons.
`/clear` resets conversational history; `/exit` releases the process and GPU memory.
The chat injects no system prompt or output-language policy. Stop other model
processes if there is insufficient free VRAM.

## Files

- `project_config.py`: generic configuration loading and artifact discovery.
- `training_utils.py`: dataset checks and the complete training workflow.
- `monitoring.py`: read-only dashboard and Trainer callback.
- `chat.py`: base/adapter terminal inference.
- `scripts/`: optional data preparation tools, including an eCFR source workflow.
- `config.example.json`: public generic configuration schema/example.
- `config.local.json`: ignored local experiment settings.
- `models/`, `data/`, `outputs/`: ignored local checkpoints, datasets, and artifacts.

Each completed run contains adapter weights/configuration/tokenizer files, training
and validation metrics, and provenance. Monitor histories are separate from model
weights. An adapter requires its original base checkpoint: required model identity
belongs in artifact metadata when publishing weights, even though public code is generic.

Publish code and cleared notebooks on GitHub. Publish tested model artifacts and
shareable datasets separately, with their provenance and applicable licenses.
