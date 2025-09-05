# =============================================================================
# POTTERVERSE AI - BACKEND SERVER (FLASK) - UPGRADED
# This version includes temperature/top-k sampling for more creative generation,
# dynamically sets the output token length based on the task prefix,
# supports switching between multiple fine-tuned models,
# and uses the correct model configuration for each set of weights.
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

# --- Create a separate config for the pretrained GPT-2 model which uses a bias ---
gpt_config_pretrained = GPT_CONFIG_124M.copy()
gpt_config_pretrained.update({"qkv_bias": True})


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
# SECTION 2: GLOBAL VARIABLES
# =============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOKENIZER = None

# =============================================================================
# SECTION 2.5: DUAL MODEL MANAGEMENT (UPGRADED)
# =============================================================================

# --- Map model names to their weight file paths ---
MODEL_WEIGHTS_PATHS = {
    "multitask": "./multitask_potter_finetuned.pth",
    "gpt2": "./harry_potter_GPT2_Finetuned.pth"
}

# --- (NEW) Map model names to their configuration dictionaries ---
MODEL_CONFIGS = {
    "multitask": GPT_CONFIG_124M,
    "gpt2": gpt_config_pretrained
}

# --- Dictionary to hold the loaded models in memory ---
MODELS = { "multitask": None, "gpt2": None }

def load_models_and_tokenizer():
    """Loads all specified models and the tokenizer into memory at startup."""
    global MODELS, TOKENIZER
    
    print("Loading tokenizer...")
    TOKENIZER = tiktoken.get_encoding("gpt2")
    
    for model_name, model_path in MODEL_WEIGHTS_PATHS.items():
        print(f"\n----- Loading model: '{model_name}' -----")
        try:
            # --- (MODIFIED) Look up the correct config for the current model ---
            config = MODEL_CONFIGS[model_name]
            model = GPTModel(config)
            print(f"Instantiating model with config: qkv_bias={config['qkv_bias']}")

            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            model.to(DEVICE)
            model.eval()
            MODELS[model_name] = model
            print(f"Model '{model_name}' loaded successfully onto {DEVICE}.")
        except FileNotFoundError:
            print(f"FATAL ERROR: Model weights for '{model_name}' not found at '{model_path}'")
            print("Please ensure the .pth file is in the same directory as this script.")
        except Exception as e:
            print(f"An error occurred while loading model '{model_name}': {e}")

# =============================================================================
# SECTION 3: TEXT GENERATION LOGIC (UPGRADED)
# =============================================================================

def generate(model, idx, max_new_tokens, context_size, temperature=0.7, top_k=50):
    """Generates text from a starting token index using a specific model instance."""
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        
        logits = logits[:, -1, :]
        if temperature > 0:
            logits = logits / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
        
        probas = torch.nn.functional.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probas, num_samples=1)
        
        if idx_next.item() == TOKENIZER.eot_token:
            break
            
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

def generate_text_from_prompt(model_to_use, model_choice, prompt, max_new_tokens=60, temperature=0.7, top_k=50):
    """Encodes a prompt, generates text using the selected model, and decodes the result."""
    if not model_to_use or not TOKENIZER:
        return "Error: Model not loaded or tokenizer not available."

    generation_prompt = f"{prompt} <|endoftext|>"
    encoded = TOKENIZER.encode(generation_prompt, allowed_special={"<|endoftext|>"})
    idx = torch.tensor(encoded, dtype=torch.long, device=DEVICE).unsqueeze(0)

    # --- (MODIFIED) Get context size from the correct config based on model_choice ---
    context_size = MODEL_CONFIGS[model_choice]["context_length"]

    output_ids = generate(
        model=model_to_use,
        idx=idx,
        max_new_tokens=max_new_tokens,
        context_size=context_size,
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
    """API endpoint to handle text generation requests, with model selection."""
    data = request.get_json()
    if not data or 'prompt' not in data:
        return jsonify({"error": "Prompt not provided"}), 400

    prompt = data['prompt']
    model_choice = data.get('model_choice', 'multitask').lower()
    
    if model_choice not in MODELS:
        return jsonify({"error": f"Invalid model choice: '{model_choice}'.", "available_models": list(MODELS.keys())}), 400

    selected_model = MODELS[model_choice]
    if selected_model is None:
        return jsonify({"error": f"Model '{model_choice}' is not available or failed to load."}), 500
        
    max_tokens = 70
    if prompt.strip().startswith("[SORTING]"):
        max_tokens = 30
    elif prompt.strip().startswith("[SPELL]"):
        max_tokens = 10
    elif prompt.strip().startswith("[CHAT]"):
        max_tokens = 80
        
    print(f"\nReceived prompt: {prompt}")
    print(f"Using model: '{model_choice}'")
    
    # --- (MODIFIED) Pass model_choice to the generation function ---
    completion = generate_text_from_prompt(
        model_to_use=selected_model,
        model_choice=model_choice, 
        prompt=prompt, 
        max_new_tokens=max_tokens
    )
    
    print(f"Generated completion: {completion}")
    return jsonify({"completion": completion})

if __name__ == '__main__':
    load_models_and_tokenizer()
    app.run(host="0.0.0.0", port=5000, debug=False)