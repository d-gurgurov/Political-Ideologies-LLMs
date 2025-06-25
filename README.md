# Political Ideologies of LLMs

This project explores the political biases of large language models (LLMs) by evaluating them on political compass-style questionnaires. We then investigate whether it's possible to **intervene** and **steer** these models' ideological outputs through internal activation manipulation.

## Overview

- **Data Collection**: Scripts to scrape and format political ideology prompts.
- **Model Evaluation**: Run LLMs on political compass questions to determine baseline ideology.
- **Probing**: Train probes to identify internal model features correlated with political orientation.
- **Intervention**: Apply vector interventions in hidden states to shift the model's responses.
- **Intervention Evaluation**: Measure the effectiveness of interventions in changing ideological outputs.

## Scripts

| Script | Purpose |
|--------|---------|
| `0-scrape.sh` / `scrape.py` | Download and preprocess political compass data |
| `1-generate.sh` / `generate.py` | Generate model outputs for the political test |
| `2-evaluate.sh` / `evaluate.py` | Analyze model ideology from responses |
| `3-probe.sh` / `probe_binary.py` | Train probes on model activations |
| `4-intervene.sh` / `intervene.py` | Apply activation-based interventions |
| `5-evaluate_intervention.sh` / `evaluate_intervene.py` | Evaluate ideological shift after intervention |

## Directory Structure

- `data/` — Input prompts, model outputs, and evaluation files  
- `prompts/` — Political compass questions and translations

## License

This project is licensed under the MIT License. See `LICENSE` for details.
