# US Federal Regulations — Fine-Tuning Experiment

This project explores whether a language model can answer questions about US
federal regulations more accurately after training on examples drawn from the
Electronic Code of Federal Regulations ([eCFR](https://www.ecfr.gov/)).

The current experiment uses selected material from Titles 1–10, with source
snapshots dated October 8, 2026. It follows the process from preparing
question–answer examples to training a model and checking its responses.

This is an experimental project. Tests have shown factual mistakes and missed
parts of multi-question prompts, so the trained model is not a reliable legal
reference. Its answers should be checked against the relevant regulations.

## What You Will Find Here

- **Training notebook:** `finetune.ipynb` walks through preparing and running
  a training session.
- **Live monitoring:** a dashboard shows how training is progressing.
- **Chat:** `chat.py` lets you try the original model or a saved adapter.
- **Data tools:** the `scripts/` folder contains tools for building, splitting,
  selecting, and checking the training examples.
- **Configuration:** `config.example.json` is the starting point for your own
  local settings. Running the project requires a compatible Python/GPU
  environment, a base model, and prepared data.

The trained adapter is available on
[Hugging Face](https://huggingface.co/huncijr/US-federal-regulations-qlora).
An adapter contains the learned changes and needs its matching base model to
run. The model page provides training details and usage information.

This repository contains the workflow code. Model weights, datasets, training
logs, and local experiment files are kept separately.
