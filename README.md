# PotterVerseAI: A Custom GPT-2 Language Model 🧙‍♂️

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)  
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)](https://pytorch.org/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

PotterVerseAI is a complete, from-scratch implementation of a GPT-2 style language model. This project covers the entire lifecycle of a large language model: **pre-training** on a custom corpus (the complete Harry Potter book series), **multi-task fine-tuning** for specific magical tasks, and **deployment** in a user-friendly Flask web application.

The model specializes in three tasks:

- **Sorting Hat**: Recommends a Hogwarts house based on a user's personality traits.  
- **Spell Generator**: Creates a spell name based on its intended effect.  
- **Character Chat**: Engages in a role-playing conversation as a specific character (e.g., Professor Snape).  

---

## ✨ Features
- **From-Scratch Implementation**: All core Transformer components (Multi-Head Attention, LayerNorm, GELU, etc.) built in PyTorch.  
- **Pre-training Pipeline**: Trained on the complete Harry Potter series.  
- **Multi-Task Fine-Tuning**: Uses task-specific prompt prefixes (`[SORTING]`, `[SPELL]`, `[CHAT]`).  
- **Command-Line Interface**: Train, fine-tune, and infer easily.  
- **Inference Engine**: Flexible text generation with temperature & top-k sampling.  
- **Web Application**: Flask-based front-end for interactive use.  

---

## 🏛️ Project Pipeline

### 1. Pre-training 📚
The model learns the language, grammar, and narrative style of the wizarding world by reading the entire Harry Potter series, forming a **general-purpose base model**.

### 2. Multi-Task Fine-Tuning 🎯
Custom datasets with **prompt prefixes** are used for fine-tuning:
- `[SORTING] I value knowledge and wit.` → *Ravenclaw!*  
- `[SPELL] Effect: To unlock a door.` → *Alohomora*  
- `[CHAT] User to Snape: What is the secret to potions?` → *The secret is precision …*  

### 3. Inference & Application 💬
The fine-tuned model can be used via:
- **CLI**: Quick text generation tests.  
- **Web App**: Flask interface for chatting with PotterVerseAI.  

---

## 🛠️ Technologies & Libraries

**Core Logic**  
- Python 3.9+  
- PyTorch  

**Data Handling & Tokenization**  
- TikToken  
- Pandas  
- PyPDF  

**Visualization & CLI**  
- Matplotlib  
- Argparse  

**Web Application**  
- Flask  

---


## Model Download Link
```bash
https://www.kaggle.com/models/deucalionsash/harrypottertrainedmodel ( Trained From Scratch Model )
https://www.kaggle.com/models/deucalionsash/finetunedmodelharrypotter ( Scratch Trained Model Finetuned )
https://www.kaggle.com/models/deucalionsash/harrypotterfinetuned_ongpt2_weights ( Model is on GPT2 weights)
```

## 🚀 Getting Started

### 1. Prerequisites
```bash
# Clone the repository
git clone https://github.com/Sshashank18/PotterVerseAI.git

# Install dependencies
pip install torch tiktoken pandas matplotlib pypdf flask tensorflow
```

### 2. Data Preparation
- **Pre-training**: Place the full Harry Potter PDF in the root directory.  
- **Fine-tuning**: Create CSV files (`sorting_hat_dataset.csv`, `spell_generator_dataset.csv`, `character_chat_dataset.csv`) with prompt-completion pairs.  

### 3. Workflows Overview ⚡
Your script supports multiple training & inference workflows. Below is a quick guide:

| Command | Description |
|---------|-------------|
| `python main.py train` | Train GPT-2 **from scratch** on the Harry Potter corpus. |
| `python main.py finetune-scratch` | Fine-tune the **scratch-trained** model on multi-task datasets. |
| `python main.py infer-scratch` | Run inference using the **scratch-trained + fine-tuned** model. |
| `python main.py pretrain-finetune` | Start from a **pretrained GPT-2 model**, then fine-tune on Harry Potter BOOK. |
| `python main.py finetune-pretrained` | Further fine-tune an already **pretrained + fine-tuned** model. |
| `python main.py infer-finetuned` | Run inference with the **pretrained + fine-tuned** model. |

---

### 4. Running the Pipeline
```bash
# Stage 1: Pre-train the Model
python main.py train

# Stage 2: Fine-tune the Model
python main.py finetune-scratch  # if trained from scratch
python main.py finetune-pretrained  # if starting from pretrained loaded weights

# Stage 3: Perform Inference
python main.py infer-scratch
python main.py infer-finetuned
```

### 5. Web Application
```bash
python app.py
```
Visit: [http://127.0.0.1:5000](http://127.0.0.1:5000) to interact with PotterVerseAI.

---

## 📜 Code Structure Overview

### Model Architecture
- `GPT_CONFIG_124M` – Hyperparameters  
- `LayerNorm`, `GELU`, `MultiHeadAttention`, `TransformerBlock` – Core building blocks  
- `GPTModel` – Full transformer model  

### Data Handling
- `GPTDatasetV1` – Pre-training dataset  
- `MultiTaskFineTuneDataset` – Fine-tuning dataset with prompt prefixes  
- `custom_collate_fn` – Handles sequence padding  

### Training & Logic
- `train_model_simple` – Pre-training loop  
- `run_finetuning` – Fine-tuning loop with scheduler  
- `generate` – Inference function  

### Main Execution
- `pretrain_model()` – Run pre-training  
- `finetune_model()` – Run fine-tuning  
- `perform_inference()` – CLI-based inference  

---

## ⚠️ Limitations & Results

During experiments with **pretrain-finetune** (using official GPT-2 weights), two additional trials were performed:

1. **Harry Potter book fine-tuning**
2. **Wizard dataset fine-tuning**

In both cases, the loss remained very high and the generated results were poor in quality due to severe resource constraints (limited compute and training budget).

This highlights the challenges of adapting pretrained GPT-2 models to highly domain-specific corpora without sufficient GPU resources.

---