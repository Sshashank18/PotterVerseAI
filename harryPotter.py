# =============================================================================
# TRAIN A HARRY POTTER LLM FROM A PRE-TRAINED GPT-2 MODEL (FIXED)
# =============================================================================
# Key fixes:
# - Force TensorFlow to CPU to prevent VRAM preallocation (root cause of 80GB spike).
# - Copy weights in-place (no replacing nn.Parameters).
# - Bool attention mask and non-persistent buffer.
# - Tie output head to token embedding.
# - Free TF session/vars promptly; gc after weight load.
# - Bug fix: finetune_on_tasks_pretrained uses input_model.
# - Robust pad_token_id for tiktoken.
# =============================================================================

import os
# ---- CRITICAL: Disable TF GPU BEFORE importing tensorflow ----
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# --------------------------------------------------------------

import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
import math
import urllib.request
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import time
import pandas as pd
import pypdf
import json
import argparse
from torch.optim.lr_scheduler import CosineAnnealingLR

import requests
from tqdm import tqdm
import numpy as np
import gc

# --- Main Model Configuration ---
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

# --- GPT Model Architecture ---

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var  = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

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
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        # Use boolean mask and don't persist it in state_dict (saves memory / clutter)
        mask = torch.triu(torch.ones(context_length, context_length, dtype=torch.bool), diagonal=1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x):
        b, T, _ = x.shape
        k = self.W_key(x).view(b, T, self.num_heads, self.head_dim).transpose(1, 2)     # [b,h,T,hd]
        q = self.W_query(x).view(b, T, self.num_heads, self.head_dim).transpose(1, 2)   # [b,h,T,hd]
        v = self.W_value(x).view(b, T, self.num_heads, self.head_dim).transpose(1, 2)   # [b,h,T,hd]

        att = q @ k.transpose(2, 3)  # [b,h,T,T]
        mask = self.mask[:T, :T]     # [T,T], bool
        att.masked_fill_(mask, float('-inf'))
        att = torch.softmax(att / (self.head_dim ** 0.5), dim=-1)
        att = self.dropout(att)
        y = (att @ v).transpose(1, 2).contiguous().view(b, T, self.d_out)
        return self.out_proj(y)

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

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        # Tie output head to token embedding (GPT-2 style)
        self.out_head.weight = self.tok_emb.weight
        self.cfg = cfg

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok = self.tok_emb(in_idx)
        pos = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = self.drop_emb(tok + pos)
        # (opt) gradient checkpoint on last 2 blocks to save memory
        for i, block in enumerate(self.trf_blocks):
            if self.training and i >= len(self.trf_blocks) - 2:
                x = torch.utils.checkpoint.checkpoint(block, x)
            else:
                x = block(x)
        x = self.final_norm(x)
        return self.out_head(x)

# --- Data ---

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        for i in range(0, len(token_ids) - max_length, stride):
            inp = token_ids[i:i + max_length]
            tgt = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(inp, dtype=torch.long))
            self.target_ids.append(torch.tensor(tgt, dtype=torch.long))
    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

# --- Generation ---

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
    return torch.tensor(encoded, dtype=torch.long).unsqueeze(0)

def token_ids_to_text(token_ids, tokenizer):
    return tokenizer.decode(token_ids.squeeze(0).tolist())

# --- Training utils ---

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch  = input_batch.to(device, non_blocking=True)
    target_batch = target_batch.to(device, non_blocking=True)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

@torch.no_grad()
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.0
    if not data_loader: return float("nan")
    num_batches = num_batches or len(data_loader)
    model.eval()
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches: break
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()
    model.train()
    return total_loss / num_batches

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
    val_loss   = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    return train_loss, val_loss

def generate_and_print_sample(model, tokenizer, device, start_context, temperature, top_k):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate(model=model, idx=encoded, max_new_tokens=50, context_size=context_size,
                             temperature=temperature, top_k=top_k)
    print(token_ids_to_text(token_ids, tokenizer).replace("\n", " "))
    model.train()

def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer, temperature, top_k,
                       accumulation_steps, save_path="harry_potter_GPT2.pth"):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    best_val_loss = float('inf')
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    for epoch in range(num_epochs):
        print(f"Started Epoch : {epoch}")
        print_gpu_memory()
        model.train()
        for i, (input_batch, target_batch) in enumerate(train_loader):
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                loss = calc_loss_batch(input_batch, target_batch, model, device)
                loss = loss / accumulation_steps
            scaler.scale(loss).backward()
            tokens_seen += input_batch.numel()

            if (i + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % eval_freq == 0:
                    train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    track_tokens_seen.append(tokens_seen)
                    print(f"Ep {epoch+1} (Step {global_step:06d}): Train {train_loss:.3f}, Val {val_loss:.3f}")

                    # ✅ Save the best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save(model.state_dict(), save_path)
                        print(f"🔸 Best model saved at step {global_step} with Val loss {val_loss:.3f}")

        # leftover grads
        if len(train_loader) % accumulation_steps != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

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

# --- Fine-Tuning dataset ---

class MultiTaskFineTuneDataset(Dataset):
    def __init__(self, csv_files, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.eot = "<|endoftext|>"
        all_data = []
        for file_path in csv_files:
            df = pd.read_csv(file_path)
            if "sorting_hat"     in file_path: df['prefix'] = "[SORTING]"
            elif "spell_generator" in file_path: df['prefix'] = "[SPELL]"
            elif "character_chat"  in file_path: df['prefix'] = "[CHAT]"
            else: df['prefix'] = ""
            all_data.append(df)
        self.data = pd.concat(all_data, ignore_index=True)

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = f"\"{row['prefix']} {row['prompt']}\" \"{self.eot} {row['completion']}\""
        token_ids = self.tokenizer.encode(text, allowed_special={self.eot})
        prompt_text = f"\"{row['prefix']} {row['prompt']}\" \"{self.eot}\""
        prompt_ids  = self.tokenizer.encode(prompt_text, allowed_special={self.eot})
        split_idx = len(prompt_ids) - 1

        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]

        input_ids  = torch.tensor(token_ids[:-1], dtype=torch.long)
        target_ids = torch.tensor(token_ids[1:],  dtype=torch.long)
        target_ids[:split_idx] = -100
        return input_ids, target_ids

def custom_collate_fn(batch, pad_token_id: int):
    inputs, targets = zip(*batch)
    max_len = max(len(s) for s in inputs)
    padded_inputs  = torch.full((len(inputs),  max_len), pad_token_id, dtype=torch.long)
    padded_targets = torch.full((len(targets), max_len), -100,        dtype=torch.long)
    for i, seq in enumerate(inputs):  padded_inputs[i, :len(seq)]  = seq
    for i, seq in enumerate(targets): padded_targets[i, :len(seq)] = seq
    return padded_inputs, padded_targets

def run_finetuning(model, data_loader, optimizer, device, num_epochs):
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    for epoch in range(num_epochs):
        total_loss = 0.0
        for input_batch, target_batch in data_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                loss = calc_loss_batch(input_batch, target_batch, model, device)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        scheduler.step()
        print(f"Epoch {epoch+1}/{num_epochs} | Average Loss: {total_loss/len(data_loader):.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

# --- GPT-2 weight download / load (TF on CPU only) ---

def download_and_load_gpt2(model_size, models_dir):
    
    import os
    # --- Disable GPU for TensorFlow only ---
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    import tensorflow as tf
    import numpy as np
    import gc
    
    
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
            response = requests.get(url, stream=True)
            response.raise_for_status()
            total_size_in_bytes = int(response.headers.get('content-length', 0))
            block_size = 1024
            progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
            with open(file_path, 'wb') as f:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data)); f.write(data)
            progress_bar.close()
            if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                print("ERROR, something went wrong during download")
        else:
            print(f"File already exists: {file_path}")

    with open(os.path.join(model_dir, 'hparams.json')) as f:
        hparams = json.load(f)

    params = []
    tf.compat.v1.disable_eager_execution()
    config = tf.compat.v1.ConfigProto(device_count={'GPU': 0})  # <<< CPU-only session
    with tf.compat.v1.Session(config=config) as sess:
        ckpt_path = tf.train.latest_checkpoint(model_dir)
        saver = tf.compat.v1.train.import_meta_graph(ckpt_path + '.meta')
        saver.restore(sess, ckpt_path)
        trained_vars = tf.compat.v1.trainable_variables()
        for var in trained_vars:
            val = var.eval()  # numpy array (float32)
            params.append({"name": var.name, "value": val})
    tf.compat.v1.reset_default_graph()
    tf.keras.backend.clear_session()

    params_dict = {}
    for param in params:
        keys = param['name'].replace(':0', '').split('/')[1:]  # drop 'model'
        cur = params_dict
        for i, key in enumerate(keys):
            if i == len(keys) - 1:
                cur[key] = param['value']
            else:
                if key not in cur: cur[key] = {}
                cur = cur[key]

    # free python lists early
    del params
    gc.collect()
    return hparams, params_dict

def _copy_param_(torch_param: torch.nn.Parameter, array):
    """Copy numpy array/torch tensor into existing nn.Parameter (dtype-preserving)."""
    if isinstance(array, np.ndarray):
        tensor = torch.from_numpy(array)
    else:
        tensor = torch.as_tensor(array)
    tensor = tensor.to(dtype=torch_param.dtype, device=torch_param.device)
    with torch.no_grad():
        torch_param.copy_(tensor)

def load_weights_into_gpt(gpt: GPTModel, params):
    # embed + pos
    _copy_param_(gpt.pos_emb.weight, np.asarray(params['wpe'], dtype=np.float32))
    _copy_param_(gpt.tok_emb.weight, np.asarray(params['wte'], dtype=np.float32))

    for b in range(len(gpt.trf_blocks)):
        p = params[f'h{b}']

        # c_attn: combined qkv
        cw = np.asarray(p["attn"]["c_attn"]["w"], dtype=np.float32)
        cb = np.asarray(p["attn"]["c_attn"]["b"], dtype=np.float32)
        q_w, k_w, v_w = np.split(cw, 3, axis=-1)
        q_b, k_b, v_b = np.split(cb, 3, axis=-1)

        block = gpt.trf_blocks[b]
        _copy_param_(block.att.W_query.weight, np.squeeze(q_w).T)
        _copy_param_(block.att.W_key.weight,   np.squeeze(k_w).T)
        _copy_param_(block.att.W_value.weight, np.squeeze(v_w).T)
        _copy_param_(block.att.W_query.bias, q_b.squeeze())
        _copy_param_(block.att.W_key.bias,   k_b.squeeze())
        _copy_param_(block.att.W_value.bias, v_b.squeeze())

        _copy_param_(block.att.out_proj.weight, np.squeeze(p["attn"]["c_proj"]["w"]).T)
        _copy_param_(block.att.out_proj.bias,   p["attn"]["c_proj"]["b"])

        _copy_param_(block.ff.layers[0].weight, np.squeeze(p["mlp"]["c_fc"]["w"]).T)
        _copy_param_(block.ff.layers[0].bias,   p["mlp"]["c_fc"]["b"])
        _copy_param_(block.ff.layers[2].weight, np.squeeze(p["mlp"]["c_proj"]["w"]).T)
        _copy_param_(block.ff.layers[2].bias,   p["mlp"]["c_proj"]["b"])

        _copy_param_(block.norm1.scale, p["ln_1"]["g"])
        _copy_param_(block.norm1.shift, p["ln_1"]["b"])
        _copy_param_(block.norm2.scale, p["ln_2"]["g"])
        _copy_param_(block.norm2.shift, p["ln_2"]["b"])

    _copy_param_(gpt.final_norm.scale, params["ln_f"]["g"])
    _copy_param_(gpt.final_norm.shift, params["ln_f"]["b"])

    # Tied output head already points to tok_emb.weight, no extra copy needed.
    # (But we can refresh once for clarity)
    _copy_param_(gpt.out_head.weight, np.asarray(params["wte"], dtype=np.float32))

    # Free big dict ASAP
    del params
    gc.collect()

# --- Workflows ---

def train_from_scratch(config, device, tokenizer):
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
    val_loader   = create_dataloader_v1(val_data,   batch_size=2, max_length=config["context_length"], stride=config["context_length"]//2, shuffle=False)

    torch.manual_seed(123)
    model = GPTModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=0.1)

    print("\n--- Starting Training from Scratch on Harry Potter Book ---")
    start_time = time.time()
    num_epochs = 12
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=20, eval_iter=5,
        start_context="The room was", tokenizer=tokenizer,
        temperature=1.0, top_k=50, accumulation_steps=8
    )
    print(f"\n--- Training from Scratch Finished in {(time.time() - start_time)/60:.2f} minutes ---")

    if train_losses:
        epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
        plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses, "scratch_train_losses.png")

    torch.save(model.state_dict(), "harry_potter_scratch.pth")
    print("Model trained from scratch saved to 'harry_potter_scratch.pth'")

def clear_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def finetune_from_pretrained_on_book(model, device, tokenizer):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    print("\n--- Starting Fine-Tuning of Pre-trained GPT-2 on Harry Potter Book 1 ---")
    torch.cuda.empty_cache()

    # Freeze all, unfreeze last 2 blocks + final_norm + head
    for p in model.parameters(): p.requires_grad = False
    for p in model.trf_blocks[-2:].parameters(): p.requires_grad = True
    for p in model.final_norm.parameters():      p.requires_grad = True
    # out_head tied to tok_emb; we only train last layers here
    # (If you want to train head too, unfreeze tok_emb because weights are tied)
    for p in model.out_head.parameters(): p.requires_grad = False

    with open("harry_potter_book_1.txt", "r", encoding="utf-8") as file:
        text_data = file.read()

    max_len = GPT_CONFIG_124M["context_length"]
    stride = max_len // 2
    batch_size = 2

    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))
    train_data, val_data = text_data[:split_idx], text_data[split_idx:]

    train_loader = create_dataloader_v1(train_data, batch_size=batch_size, max_length=max_len, stride=stride)
    val_loader   = create_dataloader_v1(val_data,   batch_size=batch_size, max_length=max_len, stride=stride, shuffle=False)

    trainable_params = [p for p in model.parameters() if p.requires_grad] # IN CASE I WANT TO TRAIN ONLY LAST LAYERS
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4,weight_decay=0.01)

    num_epochs = 120
    train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=100, eval_iter=3,
        start_context="The room was", tokenizer=tokenizer,
        temperature=0.7, top_k=50, accumulation_steps=8
    )

    torch.save(model.state_dict(), "harry_potter_GPT2_final.pth")
    print("Model saved to 'harry_potter_GPT2_final.pth'")
    return model

def print_gpu_memory():
    if torch.cuda.is_available():
        print(f"GPU allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB | reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

def finetune_on_tasks(input_model_path, output_model_path, config, device, tokenizer):
    if not os.path.exists(input_model_path):
        print(f"Model file not found at '{input_model_path}'. Please run the required preceding stage.")
        return

    TASK_CSV_FILES = ['sorting_hat_dataset.csv', 'spell_generator_dataset.csv', 'character_chat_dataset.csv']
    for file in TASK_CSV_FILES:
        if not os.path.exists(file):
            print(f"Error: Required data file '{file}' not found. Skipping task fine-tuning.")
            return

    model = GPTModel(config)
    model.load_state_dict(torch.load(input_model_path, map_location='cpu'))
    model.to(device)

    max_length = 256
    multitask_dataset = MultiTaskFineTuneDataset(TASK_CSV_FILES, tokenizer, max_length=max_length)

    # robust pad token id using tiktoken
    pad_token_id = tiktoken.get_encoding("gpt2").encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]

    finetune_loader = DataLoader(multitask_dataset, batch_size=4, shuffle=True,
                                 collate_fn=lambda b: custom_collate_fn(b, pad_token_id=pad_token_id))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    run_finetuning(model, finetune_loader, optimizer, device, num_epochs=30)

    torch.save(model.state_dict(), output_model_path)
    print(f"\nMulti-task fine-tuned model saved to '{output_model_path}'")
    return model

def finetune_on_tasks_pretrained(input_model_path, output_model_path, config, device, tokenizer):
    """Fine-tune the book-tuned GPT-2 on tasks (bug-fixed)."""
    if not os.path.exists(input_model_path):
        print(f"Model file not found at '{input_model_path}'. Please run the required preceding stage.")
        return

    TASK_CSV_FILES = ['sorting_hat_dataset.csv', 'spell_generator_dataset.csv', 'character_chat_dataset.csv']
    for file in TASK_CSV_FILES:
        if not os.path.exists(file):
            print(f"Error: Required data file '{file}' not found. Skipping task fine-tuning.")
            return

    model = GPTModel(config)
    model.load_state_dict(torch.load(input_model_path, map_location='cpu'))
    model.to(device)
    
    torch.cuda.empty_cache()

    # Freeze most layers; unfreeze last 2 + final_norm ; keep head tied
    for p in model.parameters(): p.requires_grad = False
    for p in model.trf_blocks[-2:].parameters(): p.requires_grad = True
    for p in model.final_norm.parameters():      p.requires_grad = True
    for p in model.out_head.parameters():        p.requires_grad = False  # tied to tok_emb

    TASK_CSV_FILES = ['sorting_hat_dataset.csv', 'spell_generator_dataset.csv', 'character_chat_dataset.csv']
    for file in TASK_CSV_FILES:
        if not os.path.exists(file):
            print(f"Error: Required data file '{file}' not found. Skipping task fine-tuning.")
            return

    max_length = 256
    multitask_dataset = MultiTaskFineTuneDataset(TASK_CSV_FILES, tokenizer, max_length=max_length)

    pad_token_id = tiktoken.get_encoding("gpt2").encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]

    finetune_loader = DataLoader(multitask_dataset, batch_size=4, shuffle=True,
                                 collate_fn=lambda b: custom_collate_fn(b, pad_token_id=pad_token_id))

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = torch.optim.SGD(trainable_params, lr=0.01, momentum=0.9, nesterov=True, weight_decay=1e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4,weight_decay=0.00001)

    print(f"\n--- Starting Multi-Task Fine-Tuning ---")
    run_finetuning(model, finetune_loader, optimizer, device, num_epochs=70)

    torch.save(model.state_dict(), output_model_path)
    print(f"\nMulti-task fine-tuned model saved to '{output_model_path}'")
    return model

# --- Inference ---

def perform_inference(model_path, config, tokenizer, prompt):
    if not os.path.exists(model_path):
        print(f"Model file not found at '{model_path}'. Please run the required training stage first.")
        return
    device = torch.device("cpu")
    model = GPTModel(config)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print(f"\n--- Generating text from prompt: '{prompt}' ---")
    encoded_prompt = text_to_token_ids(prompt, tokenizer).to(device)
    output_ids = generate(model=model, idx=encoded_prompt, max_new_tokens=100,
                          context_size=config["context_length"], temperature=0.7, top_k=50)
    print("\nGenerated Text:\n", token_ids_to_text(output_ids, tokenizer))

def perform_task_inference(model_path, config, tokenizer):
    if not os.path.exists(model_path):
        print(f"Final model not found at '{model_path}'. Please run the required fine-tuning stage.")
        return

    device = torch.device("cpu")
    model = GPTModel(config)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print("\n--- Testing the Final Multi-Task Fine-tuned Model ---")
    tasks_to_test = {
        "Sorting Hat":    {"prefix": "[SORTING]", "prompt": "I value knowledge and wit."},
        "Spell Generator":{"prefix": "[SPELL]",   "prompt": "Effect: To unlock a door."},
        "Character Chat": {"prefix": "[CHAT]",    "prompt": "User to Snape: What is the secret to potions?"}
    }
    for task, info in tasks_to_test.items():
        print(f"\n--- Testing Task: {task} ---")
        prompt = f"\"{info['prefix']} {info['prompt']}\" \"<|endoftext|>\""
        encoded = text_to_token_ids(prompt, tokenizer).to(device)
        output_ids = generate(model, encoded, max_new_tokens=50,
                              context_size=config["context_length"],
                              temperature=0.2, top_k=10)
        full_text = token_ids_to_text(output_ids, tokenizer)
        try:
            completion = full_text.split("<|endoftext|>")[1].strip().replace("\"", "")
        except IndexError:
            completion = "Model did not produce a valid completion."
        print(f"Input: {info['prompt']}\nCompletion: {completion}")

# --- CLI ---

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run stages of the Harry Potter GPT model pipeline.")
    parser.add_argument(
        'stage', choices=['train', 'infer-scratch', 'finetune-scratch', 'pretrain-finetune', 'finetune-pretrained', 'infer-finetuned'],
        help=(
            "'train': Train a model from scratch on the book.\n"
            "'infer-scratch': Inference on the scratch-trained model.\n"
            "'finetune-scratch': Fine-tune the scratch-trained model on tasks.\n"
            "'pretrain-finetune': Load official GPT-2 weights and fine-tune on the book.\n"
            "'finetune-pretrained': Fine-tune the book-tuned GPT-2 model on tasks.\n"
            "'infer-finetuned': Inference on the final, task-finetuned model."
        )
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = tiktoken.get_encoding("gpt2")

    gpt_config_scratch = GPT_CONFIG_124M.copy()
    gpt_config_pretrained = GPT_CONFIG_124M.copy()
    gpt_config_pretrained.update({"context_length": 1024, "qkv_bias": True})

    if args.stage == 'train':
        train_from_scratch(gpt_config_scratch, device, tokenizer)

    elif args.stage == 'infer-scratch':
        perform_inference("harry_potter_scratch.pth", gpt_config_scratch, tokenizer,
                          prompt="The last thing Harry ever saw")

    elif args.stage == 'finetune-scratch':
        finetune_on_tasks(
            input_model_path="harry_potter_scratch.pth",
            output_model_path="multitask_potter_finetuned.pth",
            config=gpt_config_scratch, device=device, tokenizer=tokenizer
        )

    elif args.stage == 'pretrain-finetune':
        model = GPTModel(gpt_config_pretrained)
        print("\n--- Initializing with Pre-trained GPT-2 (124M) Weights ---")
        _, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2_124M")
        load_weights_into_gpt(model, params)
        clear_memory()
        model.to(device)
        print_gpu_memory()
        finetune_from_pretrained_on_book(model, device, tokenizer)

    elif args.stage == 'finetune-pretrained':
        finetune_on_tasks_pretrained(
            input_model_path="harry_potter_GPT2.pth",
            output_model_path="harry_potter_GPT2_Finetuned.pth",
            config=gpt_config_pretrained, device=device, tokenizer=tokenizer
        )

    elif args.stage == 'infer-finetuned':
        perform_task_inference(
            model_path="multitask_potter_finetuned.pth",
            config=GPT_CONFIG_124M,
            tokenizer=tokenizer
        )
