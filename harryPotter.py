# =============================================================================
# TRAIN A HARRY POTTER LLM FROM A PRE-TRAINED GPT-2 MODEL
# =============================================================================
# This script orchestrates multiple distinct workflows for training and
# fine-tuning a GPT model on Harry Potter-themed data.
#
# Workflows available via command-line arguments:
# 1. 'train': Train a model from scratch on the Harry Potter book.
# 2. 'infer-scratch': Run inference on the model from workflow 1.
# 3. 'finetune-scratch': Fine-tune the scratch-trained model on specific tasks.
# 4. 'pretrain-finetune': Load official GPT-2 weights and fine-tune on the book.
# 5. 'finetune-pretrained': Fine-tune the book-tuned GPT-2 model on tasks.
# 6. 'infer-finetuned': Run inference on the final, multi-task-tuned model.
#
# Requirements:
# pip install torch tiktoken pypdf pandas matplotlib tensorflow tqdm
#
# Required data files for fine-tuning stages (must be in the same directory):
# - sorting_hat_dataset.csv
# - spell_generator_dataset.csv
# - character_chat_dataset.csv
# =============================================================================

import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
import math
import os
import urllib.request
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import time
import pandas as pd
import pypdf
import json
import shutil
import argparse
from torch.optim.lr_scheduler import CosineAnnealingLR

# gpt_download.py
import requests
from tqdm import tqdm
import tensorflow as tf
import numpy as np

def download_and_load_gpt2(model_size, models_dir):
    """
    Downloads the GPT-2 model weights and configuration files from OpenAI's
    servers and loads them into memory using TensorFlow.
    """
    if model_size not in ('124M', '355M', '774M', '1558M'):
        raise ValueError(f"Model size not in ('124M', '355M', '774M', '1558M'): {model_size}")

    model_dir = os.path.join(models_dir, model_size)
    os.makedirs(model_dir, exist_ok=True)

    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    files = [
        "checkpoint", "encoder.json", "hparams.json",
        "model.ckpt.data-00000-of-00001", "model.ckpt.index", "model.ckpt.meta",
        "vocab.bpe"
    ]

    for file in files:
        file_path = os.path.join(model_dir, file)
        if not os.path.exists(file_path):
            url = f"{base_url}/{model_size}/{file}"
            print(f"Downloading {url}")
            # Use requests with stream=True for progress bar
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Will raise an HTTPError for bad responses
            total_size_in_bytes = int(response.headers.get('content-length', 0))
            block_size = 1024  # 1 Kibibyte
            
            progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
            with open(file_path, 'wb') as f:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    f.write(data)
            progress_bar.close()

            if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                print("ERROR, something went wrong during download")
        else:
            print(f"File already exists: {file_path}")

    # Load hparams
    with open(os.path.join(model_dir, 'hparams.json')) as f:
        hparams = json.load(f)

    # Load params
    params = []
    # Use tf.compat.v1 for compatibility with older TensorFlow checkpoint formats
    tf.compat.v1.disable_eager_execution()
    with tf.compat.v1.Session() as sess:
        ckpt_path = tf.train.latest_checkpoint(model_dir)
        saver = tf.compat.v1.train.import_meta_graph(ckpt_path + '.meta')
        saver.restore(sess, ckpt_path)
        
        # Collect all trainable variables
        trained_vars = tf.compat.v1.trainable_variables()
        for var in trained_vars:
            params.append({
                "name": var.name,
                "value": var.eval()
            })
    
    # Reset the default graph to avoid issues with subsequent TF operations
    tf.compat.v1.reset_default_graph()

    # Organize params into a more accessible dictionary
    params_dict = {}
    for param in params:
        # e.g., model/h0/attn/c_attn/w:0 -> ['model', 'h0', 'attn', 'c_attn', 'w']
        keys = param['name'].replace(':0', '').split('/')[1:]
        
        current_level = params_dict
        for i, key in enumerate(keys):
            if i == len(keys) - 1:
                current_level[key] = param['value']
            else:
                if key not in current_level:
                    current_level[key] = {}
                current_level = current_level[key]

    return hparams, params_dict

# --- Main Model Configuration ---
GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias
}

# --- GPT Model Architecture Classes ---

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))
        ))

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            nn.GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)
        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        return context_vec

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"], d_out=cfg["emb_dim"],
            context_length=cfg["context_length"], num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"], qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x

# Add this to your model definition
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        # Use gradient checkpointing
        self.trf_blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        self.cfg = cfg

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        
        # Use gradient checkpointing for transformer blocks
        for i, block in enumerate(self.trf_blocks):
            if self.training and i >= len(self.trf_blocks) - 2:  # Only checkpoint last 2 blocks
                x = torch.utils.checkpoint.checkpoint(block, x)
            else:
                x = block(x)
        
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
    
# --- Data Loading Classes ---

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


# --- Text Generation Utilities ---

def generate(model, idx, max_new_tokens, context_size, temperature=1.0, top_k=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        if temperature > 0.0:
            logits = logits / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    return torch.tensor(encoded).unsqueeze(0)

def token_ids_to_text(token_ids, tokenizer):
    return tokenizer.decode(token_ids.squeeze(0).tolist())


# --- Training, Evaluation, and Plotting Utilities ---

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    
    with torch.cuda.amp.autocast():  # Use mixed precision
        logits = model(input_batch)
        loss = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1), target_batch.flatten()
        )
    
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if not data_loader: return float("nan")
    num_batches = num_batches or len(data_loader)
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches: break
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()
    return total_loss / num_batches

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

def generate_and_print_sample(model, tokenizer, device, start_context, temperature, top_k):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate(
            model=model, idx=encoded, max_new_tokens=50, context_size=context_size,
            temperature=temperature, top_k=top_k
        )
    print(token_ids_to_text(token_ids, tokenizer).replace("\n", " "))
    model.train()

def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer, temperature, top_k,
                       accumulation_steps):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    
    # Use gradient scaling for mixed precision
    scaler = torch.cuda.amp.GradScaler()
    
    for epoch in range(num_epochs):
        print(f"Started Epoch : {epoch}")
        model.train()
        print_gpu_memory()
        
        for i, (input_batch, target_batch) in enumerate(train_loader):
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast():
                loss = calc_loss_batch(input_batch, target_batch, model, device)
                loss = loss / accumulation_steps
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            tokens_seen += input_batch.numel()
            
            # Perform optimizer step after accumulating gradients
            if (i + 1) % accumulation_steps == 0:
                # Unscale gradients
                scaler.unscale_(optimizer)
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                global_step += 1
                
                # Evaluation logic
                if global_step % eval_freq == 0:
                    train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    track_tokens_seen.append(tokens_seen)
                    print(f"Ep {epoch+1} (Step {global_step:06d}): Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
        
        # Handle remaining gradients
        if len(train_loader) % accumulation_steps != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
        # Generate sample
        generate_and_print_sample(model, tokenizer, device, start_context, temperature, top_k)
        
    return train_losses, val_losses, track_tokens_seen

def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses, plot_name="losses.png"):
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs"); ax1.set_ylabel("Loss"); ax1.legend()
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    plt.savefig(plot_name)
    plt.show()


# --- Fine-Tuning Classes and Functions ---

class MultiTaskFineTuneDataset(Dataset):
    def __init__(self, csv_files, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.end_of_text_token = "<|endoftext|>"
        all_data = []
        for file_path in csv_files:
            df = pd.read_csv(file_path)
            if "sorting_hat" in file_path: df['prefix'] = "[SORTING]"
            elif "spell_generator" in file_path: df['prefix'] = "[SPELL]"
            elif "character_chat" in file_path: df['prefix'] = "[CHAT]"
            else: df['prefix'] = ""
            all_data.append(df)
        self.data = pd.concat(all_data, ignore_index=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = f"\"{row['prefix']} {row['prompt']}\" \"{self.end_of_text_token} {row['completion']}\""

        token_ids = self.tokenizer.encode(text, allowed_special={self.end_of_text_token})
        prompt_text = f"\"{row['prefix']} {row['prompt']}\" \"{self.end_of_text_token}\""
        prompt_ids = self.tokenizer.encode(prompt_text, allowed_special={self.end_of_text_token})
        split_idx = len(prompt_ids) - 1

        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        
        input_ids = torch.tensor(token_ids[:-1], dtype=torch.long)
        target_ids = torch.tensor(token_ids[1:], dtype=torch.long)
        target_ids[:split_idx] = -100
        
        return input_ids, target_ids

def custom_collate_fn(batch, pad_token_id):
    inputs, targets = zip(*batch)
    max_len = max(len(s) for s in inputs)
    padded_inputs = torch.full((len(inputs), max_len), pad_token_id, dtype=torch.long)
    padded_targets = torch.full((len(targets), max_len), -100, dtype=torch.long)
    for i, seq in enumerate(inputs): padded_inputs[i, :len(seq)] = seq
    for i, seq in enumerate(targets): padded_targets[i, :len(seq)] = seq
    return padded_inputs, padded_targets

def run_finetuning(model, data_loader, optimizer, device, num_epochs):
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    model.train()
    scaler = torch.cuda.amp.GradScaler()
    
    for epoch in range(num_epochs):
        total_loss = 0
        for input_batch, target_batch in data_loader:
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                loss = calc_loss_batch(input_batch, target_batch, model, device)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
        
        scheduler.step()
        avg_loss = total_loss / len(data_loader)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{num_epochs} | Average Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")


# --- Weight Loading Utilities for GPT-2 ---

def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))

def load_weights_into_gpt(gpt, params):
    # This function is now corrected to match the flat structure of the downloaded weights

    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])

    for b in range(len(gpt.trf_blocks)):
        # Access each block's parameters using the 'h{b}' format directly from params
        block_params = params[f'h{b}']
        
        # Split the combined QKV weights and remove singleton dimensions
        q_w, k_w, v_w = np.split(
            (block_params["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, np.squeeze(q_w).T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, np.squeeze(k_w).T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, np.squeeze(v_w).T)

        # Split the combined QKV biases
        q_b, k_b, v_b = np.split(
            (block_params["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)

        # Apply squeeze to other weight matrices
        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            np.squeeze(block_params["attn"]["c_proj"]["w"]).T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            block_params["attn"]["c_proj"]["b"])

        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            np.squeeze(block_params["mlp"]["c_fc"]["w"]).T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias,
            block_params["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            np.squeeze(block_params["mlp"]["c_proj"]["w"]).T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            block_params["mlp"]["c_proj"]["b"])

        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale,
            block_params["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift,
            block_params["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale,
            block_params["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift,
            block_params["ln_2"]["b"])

    # MODIFIED: Access final layer norm parameters from the 'ln_f' dictionary
    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["ln_f"]["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["ln_f"]["b"])
    
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])
    
    torch.cuda.empty_cache()
    
    
def setup_finetuning(model):
    """Freeze all parameters except very specific layers"""
    # First, freeze ALL parameters
    
    for param in model.parameters():
        param.requires_grad = False
    
    # Only unfreeze the output head and final layer norm
    for param in model.final_norm.parameters():
        param.requires_grad = True
    
    # Unfreeze output head
    for param in model.out_head.parameters():
        param.requires_grad = True
    
    # Optionally unfreeze only the last transformer block's attention
    # for param in model.trf_blocks[-1].att.parameters():
    #     param.requires_grad = True
    
    # Print which parameters are trainable
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.2f}%)")
    
    return model

# --- Main Operational Functions for Each Stage ---

def train_from_scratch(config, device, tokenizer):
    """
    WORKFLOW 1: Train a model from scratch on the Harry Potter book.
    """
    # --- Data Loading ---
    file_path_txt = "harry_potter_book_1.txt"
    if not os.path.exists(file_path_txt):
        print(f"Text file '{file_path_txt}' not found, downloading and extracting...")
        file_path_pdf = "harry_potter_complete_series.pdf"
        url = "https://kvongcmehsanalibrary.wordpress.com/wp-content/uploads/2021/07/harrypotter.pdf"
        if not os.path.exists(file_path_pdf):
            urllib.request.urlretrieve(url, file_path_pdf)
        with open(file_path_pdf, 'rb') as f:
            pdf_reader = pypdf.PdfReader(f)
            text_data = "".join(page.extract_text() or "" for page in pdf_reader.pages[4:234]) # Book 1
        text_data = text_data.replace(" \n", " ").replace("\n", " ")
        with open(file_path_txt, "w", encoding="utf-8") as file:
            file.write(text_data)
        print(f"Text extracted to '{file_path_txt}'.")
    else:
        with open(file_path_txt, "r", encoding="utf-8") as file:
            text_data = file.read()

    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))
    train_data, val_data = text_data[:split_idx], text_data[split_idx:]
    
    train_loader = create_dataloader_v1(train_data, batch_size=2, max_length=config["context_length"], stride=config["context_length"]//2)
    val_loader = create_dataloader_v1(val_data, batch_size=2, max_length=config["context_length"], stride=config["context_length"]//2, shuffle=False)
    
    # --- Model Initialization ---
    torch.manual_seed(123)
    model = GPTModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

    # --- Training ---
    print("\n--- Starting Training from Scratch on Harry Potter Book ---")
    start_time = time.time()
    num_epochs = 12
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=20, eval_iter=5,
        start_context="The room was", tokenizer=tokenizer,
        temperature=1.0, top_k=50,accumulation_steps=8
    )
    print(f"\n--- Training from Scratch Finished in {(time.time() - start_time)/60:.2f} minutes ---")

    if train_losses:
        epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
        plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses, "scratch_train_losses.png")
    
    # --- Save Model ---
    torch.save(model.state_dict(), "harry_potter_scratch.pth")
    print("Model trained from scratch saved to 'harry_potter_scratch.pth'")

def clear_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def finetune_from_pretrained_on_book(model, device, tokenizer):
    """WORKFLOW 4: Fine-tune a pre-trained GPT-2 model on the Harry Potter book."""
    print("\n--- Starting Fine-Tuning of Pre-trained GPT-2 on Harry Potter Book 1 ---")
    
    # Clear memory
    torch.cuda.empty_cache()
    
    # Setup layer freezing - freeze more layers to save memory
    for param in model.parameters():
        param.requires_grad = False
    
    # Only unfreeze the very last layers
    for param in model.trf_blocks[-2:].parameters():  # Last 2 blocks
        param.requires_grad = True
    for param in model.final_norm.parameters():
        param.requires_grad = True
    for param in model.out_head.parameters():
        param.requires_grad = True
    
    # Load data with smaller context length
    with open("harry_potter_book_1.txt", "r", encoding="utf-8") as file:
        text_data = file.read()
    
    # Use smaller context length
    max_len = 128
    stride = max_len // 2
    batch_size = 2
    
    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))
    train_data, val_data = text_data[:split_idx], text_data[split_idx:]
    
    train_loader = create_dataloader_v1(train_data, batch_size=batch_size, max_length=max_len, stride=stride)
    val_loader = create_dataloader_v1(val_data, batch_size=batch_size, max_length=max_len, stride=stride, shuffle=False)
    
    # Optimizer with weight decay only on unfrozen parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        trainable_params, 
        lr=0.1,            # Much higher learning rate for SGD
        momentum=0.9,
        nesterov=True,     # Nesterov momentum often works better
        weight_decay=1e-4
    )

    # optimizer = torch.optim.AdamW(
    #     trainable_params, 
    #     lr=1e-4,           # Lower learning rate for AdamW
    #     weight_decay=0.01,
    #     fused=True,         # Uses fused CUDA kernels (saves memory)
    #     foreach=False,      # Disable foreach implementation (saves memory)
    #     capturable=True     # For better CUDA graph capture
    # )
    
    num_epochs = 11 # Reduced epochs
    accumulation_steps = 8
    
    start_time = time.time()
    
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=100, eval_iter=3,  # Reduced evaluation frequency
        start_context="The room was", tokenizer=tokenizer,
        temperature=0.7, top_k=50,
        accumulation_steps=accumulation_steps
    )
    
    end_time = time.time()
    print(f"--- Book Fine-Tuning Finished in {(end_time - start_time)/60:.2f} minutes ---")
    
    torch.save(model.state_dict(), "harry_potter_GPT2.pth")
    print("Model saved to 'harry_potter_GPT2.pth'")
    
    return model

def print_gpu_memory():
    if torch.cuda.is_available():
        print(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"GPU Memory cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")


def finetune_on_tasks(input_model_path, output_model_path, config, device, tokenizer):
    """
    WORKFLOW 3 & 5: Fine-tune a model on specific tasks.
    """
    if not os.path.exists(input_model_path):
        print(f"Model file not found at '{input_model_path}'. Please run the required preceding stage.")
        return

    TASK_CSV_FILES = [
        'sorting_hat_dataset.csv', 'spell_generator_dataset.csv', 'character_chat_dataset.csv'
    ]
    
    for file in TASK_CSV_FILES:
        if not os.path.exists(file):
            print(f"Error: Required data file '{file}' not found. Skipping task fine-tuning.")
            return

    model = GPTModel(config)
    model.load_state_dict(torch.load(input_model_path, map_location=device))
    model.to(device)

    max_length = 256
    multitask_dataset = MultiTaskFineTuneDataset(TASK_CSV_FILES, tokenizer, max_length=max_length)
    
    pad_token_id = tokenizer.eot_token
    finetune_loader = DataLoader(
        multitask_dataset, batch_size=4, shuffle=True,
        collate_fn=lambda b: custom_collate_fn(b, pad_token_id=pad_token_id)
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    num_epochs = 30
    
    print(f"\n--- Starting Multi-Task Fine-Tuning for {num_epochs} epochs ---")
    run_finetuning(model, finetune_loader, optimizer, device, num_epochs=num_epochs)

    torch.save(model.state_dict(), output_model_path)
    print(f"\nMulti-task fine-tuned model saved to '{output_model_path}'")
    return model

def finetune_on_tasks_pretrained(input_model, output_model_path, config, device, tokenizer):
    """
    WORKFLOW 3 & 5: Fine-tune a model on specific tasks.
    """
    torch.cuda.empty_cache()
    
    # Setup layer freezing - freeze more layers to save memory
    for param in model.parameters():
        param.requires_grad = False
    
    # Only unfreeze the very last layers
    for param in model.trf_blocks[-2:].parameters():  # Last 2 blocks
        param.requires_grad = True
    for param in model.final_norm.parameters():
        param.requires_grad = True
    for param in model.out_head.parameters():
        param.requires_grad = True

    TASK_CSV_FILES = [
        'sorting_hat_dataset.csv', 'spell_generator_dataset.csv', 'character_chat_dataset.csv'
    ]
    
    for file in TASK_CSV_FILES:
        if not os.path.exists(file):
            print(f"Error: Required data file '{file}' not found. Skipping task fine-tuning.")
            return

#    model = GPTModel(config)
#    model.load_state_dict(torch.load(input_model_path, map_location=device))
#    model.to(device)

    max_length = 128
    multitask_dataset = MultiTaskFineTuneDataset(TASK_CSV_FILES, tokenizer, max_length=max_length)
    
    pad_token_id = tokenizer.eot_token
    finetune_loader = DataLoader(
        multitask_dataset, batch_size=4, shuffle=True,
        collate_fn=lambda b: custom_collate_fn(b, pad_token_id=pad_token_id)
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        trainable_params, 
        lr=0.01,            # Much higher learning rate for SGD
        momentum=0.9,
        nesterov=True,     # Nesterov momentum often works better
        weight_decay=1e-4
    )
    num_epochs = 120
    
    print(f"\n--- Starting Multi-Task Fine-Tuning for {num_epochs} epochs ---")
    run_finetuning(model, finetune_loader, optimizer, device, num_epochs=num_epochs)

    torch.save(model.state_dict(), output_model_path)
    print(f"\nMulti-task fine-tuned model saved to '{output_model_path}'")
    return model

def perform_inference(model_path, config, tokenizer, prompt):
    """
    WORKFLOW 2: Run inference using a trained-from-scratch model.
    """
    if not os.path.exists(model_path):
        print(f"Model file not found at '{model_path}'. Please run the required training stage first.")
        return
        
    device = torch.device("cpu")
    model = GPTModel(config)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print(f"\n--- Generating text from prompt: '{prompt}' ---")
    
    encoded_prompt = text_to_token_ids(prompt, tokenizer).to(device)
    output_ids = generate(
        model=model, idx=encoded_prompt, max_new_tokens=100,
        context_size=config["context_length"], temperature=0.7, top_k=50
    )
    print("\nGenerated Text:\n", token_ids_to_text(output_ids, tokenizer))

def perform_task_inference(model_path, config, tokenizer):
    """
    WORKFLOW 6: Specialized inference for the multi-task model.
    """
    if not os.path.exists(model_path):
        print(f"Final model not found at '{model_path}'. Please run the required fine-tuning stage.")
        return

    device = torch.device("cpu")
    model = GPTModel(config)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print("\n--- Testing the Final Multi-Task Fine-tuned Model ---")
    tasks_to_test = {
        "Sorting Hat": {"prefix": "[SORTING]", "prompt": "I value knowledge and wit."},
        "Spell Generator": {"prefix": "[SPELL]", "prompt": "Effect: To unlock a door."},
        "Character Chat": {"prefix": "[CHAT]", "prompt": "User to Snape: What is the secret to potions?"}
    }
    
    for task, info in tasks_to_test.items():
        print(f"\n--- Testing Task: {task} ---")
        prompt = f"\"{info['prefix']} {info['prompt']}\" \"<|endoftext|>\""
        encoded = text_to_token_ids(prompt, tokenizer).to(device)
        
        output_ids = generate(
            model, encoded, max_new_tokens=50, 
            context_size=config["context_length"], 
            temperature=0.2, top_k=10
        )
        
        full_text = token_ids_to_text(output_ids, tokenizer)
        try:
            completion = full_text.split("<|endoftext|>")[1].strip().replace("\"", "")
        except IndexError:
            completion = "Model did not produce a valid completion."
            
        print(f"Input: {info['prompt']}\nCompletion: {completion}")

# --- Command-Line Interface ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run stages of the Harry Potter GPT model pipeline.")
    parser.add_argument(
        'stage', choices=['train', 'infer-scratch', 'finetune-scratch', 'pretrain-finetune', 'finetune-pretrained', 'infer-finetuned'],
        help=(
            "The stage to run:\n"
            "'train': Train a model from scratch on the book.\n"
            "'infer-scratch': Inference on the scratch-trained model.\n"
            "'finetune-scratch': Fine-tune the scratch-trained model on tasks.\n"
            "'pretrain-finetune': Load GPT-2 weights and fine-tune on the book.\n"
            "'finetune-pretrained': Fine-tune the book-tuned GPT-2 model on tasks.\n"
            "'infer-finetuned': Inference on the final, task-finetuned model."
        )
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    tokenizer = tiktoken.get_encoding("gpt2")

    # Config for scratch model
    gpt_config_scratch = GPT_CONFIG_124M.copy()
    
    # Config for pre-trained model (needs bias and correct context length)
    gpt_config_pretrained = GPT_CONFIG_124M.copy()
    gpt_config_pretrained.update({"context_length": 1024, "qkv_bias": True})

    if args.stage == 'train':
        # WORKFLOW 1
        train_from_scratch(gpt_config_scratch, device, tokenizer)
    
    elif args.stage == 'infer-scratch':
        # WORKFLOW 2
        perform_inference(
            model_path="harry_potter_scratch.pth",
            config=gpt_config_scratch,
            tokenizer=tokenizer,
            prompt="The last thing Harry ever saw"
        )

    elif args.stage == 'finetune-scratch':
        # WORKFLOW 3
        finetune_on_tasks(
            input_model_path="harry_potter_scratch.pth",
            output_model_path="multitask_potter_finetuned.pth",
            config=gpt_config_scratch,
            device=device,
            tokenizer=tokenizer
        )
    
    elif args.stage == 'pretrain-finetune':
        # WORKFLOW 4
        model = GPTModel(gpt_config_pretrained)
        print("\n--- Initializing with Pre-trained GPT-2 (124M) Weights ---")
        settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2_124M")
        load_weights_into_gpt(model, params)
        clear_memory()
        model.to(device)
        finetune_from_pretrained_on_book(model, device, tokenizer)

    elif args.stage == 'finetune-pretrained':
        # WORKFLOW 5
        model = GPTModel(gpt_config_pretrained)
        print("\n--- Initializing with Pre-trained GPT-2 (124M) Weights ---")
        settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2_124M")
        load_weights_into_gpt(model, params)
        clear_memory()
        model.to(device)
        finetune_on_tasks_pretrained(
            input_model=model,
            output_model_path="harry_potter_GPT2_Finetuned.pth",
            config=gpt_config_pretrained,
            device=device,
            tokenizer=tokenizer
        )

    elif args.stage == 'infer-finetuned':
        # WORKFLOW 6
        perform_task_inference(
#           model_path="harry_potter_GPT2_Finetuned.pth",
            model_path="multitask_potter_finetuned.pth",
#            config=gpt_config_pretrained,
	       config=GPT_CONFIG_124M,
            tokenizer=tokenizer
        )