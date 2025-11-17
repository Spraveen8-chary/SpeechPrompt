# PromptSpeech
Unified Prompting Framework for Speech Language Models

## Overview
This repository reproduces the IEEE TASLP 2024 paper:
**"SpeechPrompt: Prompting Speech Language Models for Speech Processing Tasks"**

### Project Components
- Speech Quantization (HuBERT/mHuBERT + KMeans)
- Prompt-Tuned Speech LMs (GSLM / Unit mBART)
- Fixed & Learnable Verbalizers
- Unified Speech-to-Unit Reformulation
- Few-Shot and Speech Generation Pipelines

### How to Run
```bash
python main.py --mode prepare
python main.py --mode train
python main.py --mode eval
python main.py --mode generate
```
