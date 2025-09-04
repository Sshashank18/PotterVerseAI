# =============================================================================
# POTTERVERSE AI - BACKEND SERVER (FLASK) - UPGRADED
# This version includes temperature/top-k sampling for more creative generation
# and dynamically sets the output token length based on the task prefix.
# =============================================================================

import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
import os
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- Suppress deprecation warnings for a cleaner console ---
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# SECTION 1: MODEL ARCHITECTURE
# This MUST be the exact same architecture used for training and fine-tuning.
# =============================================================================

GPT_CONFIG_124M = {
    "vocab_size": 50257, "context_length": 1024, "emb_dim": 768,
    "n_heads": 12, "n_layers": 12, "drop_rate": 0.1, "qkv_bias": False
}

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
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]), GELU(), nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]))
    def forward(self, x):
        return self.layers(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.d_out, self.num_heads = d_out, num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys, queries, values = self.W_key(x), self.W_query(x), self.W_value(x)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = (attn_weights @ values).transpose(1, 2).contiguous().view(b, num_tokens, self.d_out)
        return self.out_proj(context_vec)

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(d_in=cfg["emb_dim"], d_out=cfg["emb_dim"], context_length=cfg["context_length"], num_heads=cfg["n_heads"], dropout=cfg["drop_rate"], qkv_bias=cfg["qkv_bias"])
        self.ff, self.norm1, self.norm2 = FeedForward(cfg), LayerNorm(cfg["emb_dim"]), LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])
    def forward(self, x):
        shortcut = x; x = self.norm1(x); x = self.att(x); x = self.drop_shortcut(x); x = x + shortcut
        shortcut = x; x = self.norm2(x); x = self.ff(x); x = self.drop_shortcut(x); x = x + shortcut
        return x

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds; x = self.drop_emb(x); x = self.trf_blocks(x); x = self.final_norm(x)
        return self.out_head(x)

# =============================================================================
# SECTION 2: GLOBAL VARIABLES AND MODEL LOADING
# =============================================================================

MODEL_WEIGHTS_PATH = "./multitask_potter_finetuned.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = None
TOKENIZER = None

def load_model_and_tokenizer():
    """Loads the fine-tuned model and tokenizer into memory."""
    global MODEL, TOKENIZER
    
    print("Loading model and tokenizer...")
    TOKENIZER = tiktoken.get_encoding("gpt2")
    
    MODEL = GPTModel(GPT_CONFIG_124M)
    try:
        MODEL.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE))
        print(DEVICE)
        MODEL.to(DEVICE)
        MODEL.eval()  # Set model to evaluation mode
        print(f"Model loaded successfully onto {DEVICE}.")
    except FileNotFoundError:
        print(f"FATAL ERROR: Model weights not found at '{MODEL_WEIGHTS_PATH}'")
        print("Please ensure the .pth file is in the same directory as this script.")
        exit()

# =============================================================================
# SECTION 3: TEXT GENERATION LOGIC (UPGRADED)
# =============================================================================

def generate(model, idx, max_new_tokens, context_size, temperature=0.7, top_k=50):
    """
    Generates text from a starting token index using temperature and top-k sampling.
    """
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        
        logits = logits[:, -1, :]

        # Apply temperature scaling
        if temperature > 0:
            logits = logits / temperature
            # Apply top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
        
        probas = torch.nn.functional.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probas, num_samples=1)
        
        if idx_next.item() == TOKENIZER.eot_token:
            break
            
        idx = torch.cat((idx, idx_next), dim=1)
    
    return idx

def generate_text_from_prompt(prompt, max_new_tokens=60, temperature=0.7, top_k=50):
    """Encodes a prompt, generates text, and decodes the result."""
    if not MODEL or not TOKENIZER:
        return "Error: Model not loaded."

    generation_prompt = f"{prompt} <|endoftext|>"
    
    encoded = TOKENIZER.encode(generation_prompt, allowed_special={"<|endoftext|>"})
    idx = torch.tensor(encoded, dtype=torch.long, device=DEVICE).unsqueeze(0)

    output_ids = generate(
        model=MODEL,
        idx=idx,
        max_new_tokens=max_new_tokens,
        context_size=GPT_CONFIG_124M["context_length"],
        temperature=temperature,
        top_k=top_k
    )

    full_text = TOKENIZER.decode(output_ids.squeeze(0).tolist())
    
    try:
        completion = full_text.split("<|endoftext|>")[1].strip()
        return completion
    except IndexError:
        return "The model did not produce a valid completion."

# =============================================================================
# SECTION 4: FLASK API SERVER (UPGRADED)
# =============================================================================

app = Flask(__name__)
CORS(app) 

@app.route('/generate', methods=['POST'])
def handle_generation():
    """API endpoint to handle text generation requests."""
    data = request.get_json()
    if not data or 'prompt' not in data:
        return jsonify({"error": "Prompt not provided"}), 400

    prompt = data['prompt']
    
    # --- Dynamic Token Length based on Prefix ---
    max_tokens = 70 # Default length
    if prompt.strip().startswith("[SORTING]"):
        max_tokens = 30
        print(f"Task detected: [SORTING]. Setting max_tokens to {max_tokens}.")
    elif prompt.strip().startswith("[SPELL]"):
        max_tokens = 10
        print(f"Task detected: [SPELL]. Setting max_tokens to {max_tokens}.")
    elif prompt.strip().startswith("[CHAT]"):
        max_tokens = 80
        print(f"Task detected: [CHAT]. Setting max_tokens to {max_tokens}.")
    else:
        print(f"No task prefix detected. Using default max_tokens: {max_tokens}.")
        
    print(f"\nReceived prompt: {prompt}")
    
    # Call the generation function with the dynamic token length
    completion = generate_text_from_prompt(prompt, max_new_tokens=max_tokens)
    
    print(f"Generated completion: {completion}")
    
    return jsonify({"completion": completion})

if __name__ == '__main__':
    load_model_and_tokenizer()
    app.run(host="0.0.0.0", port=5000, debug=False) # debug=False is better for 'production'

