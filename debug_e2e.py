"""
Debug E2E finetuning NaN issue.
Run: python debug_e2e.py --hf_path hf/qwen3_8b_2bit --base_model /models/Qwen/Qwen3-8B-Base
"""
import argparse
import torch
from lib.utils.unsafe_import import model_from_hf_path
from lib.linear.quantized_linear import QuantizedLinear
from torch import nn

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument('--hf_path', type=str, required=True)
parser.add_argument('--base_model', type=str, required=True)
args = parser.parse_args()

print("=" * 60)
print("1. Loading quantized model to CPU")
print("=" * 60)
model, model_str = model_from_hf_path(args.hf_path, device_map={'': 'cpu'})
print(f"Model loaded. model_str={model_str}")

# Check q_norm/k_norm weights
layer0 = model.model.layers[0]
if hasattr(layer0.self_attn, 'q_norm'):
    qn = layer0.self_attn.q_norm.weight
    kn = layer0.self_attn.k_norm.weight
    print(f"q_norm weight: shape={qn.shape} min={qn.min():.4f} max={qn.max():.4f} mean={qn.float().mean():.4f}")
    print(f"k_norm weight: shape={kn.shape} min={kn.min():.4f} max={kn.max():.4f} mean={kn.float().mean():.4f}")
    print(f"q_norm all ones: {(qn == 1).all().item()}")
    print(f"k_norm all ones: {(kn == 1).all().item()}")
else:
    print("WARNING: q_norm NOT found!")

print()
print("=" * 60)
print("2. Convert to float32 and move to GPU (like E2E finetune)")
print("=" * 60)
model = model.float().to('cuda:0')
print("Model on cuda:0 in float32")

print()
print("=" * 60)
print("3. Set QuantizedLinear modes (like E2E finetune with --ft_train_lut)")
print("=" * 60)
for name, module in model.named_modules():
    if isinstance(module, QuantizedLinear):
        module.SU = nn.Parameter(module.SU.float(), requires_grad=True)
        module.SV = nn.Parameter(module.SV.float(), requires_grad=True)
        if module.tlut is not None:
            module.tlut.requires_grad = True
        module.mode = 'train-recons'
print("All QuantizedLinear set to train-recons mode")

print()
print("=" * 60)
print("4. Single forward pass test")
print("=" * 60)
import transformers
tokenizer = transformers.AutoTokenizer.from_pretrained(args.base_model)
test_text = "The capital of France is"
tokens = tokenizer(test_text, return_tensors='pt')
input_ids = tokens.input_ids.to('cuda:0')
print(f"Input: '{test_text}' tokens={input_ids.shape}")

# Forward pass without autocast first
print("\n--- Forward WITHOUT autocast ---")
try:
    output = model(input_ids, use_cache=False)
    logits = output['logits'] if isinstance(output, dict) else output[0]
    print(f"Logits shape: {logits.shape}")
    print(f"Logits: min={logits.min():.4f} max={logits.max():.4f} mean={logits.mean():.4f}")
    print(f"Has NaN: {logits.isnan().any().item()}")
    print(f"Has Inf: {logits.isinf().any().item()}")
except Exception as e:
    print(f"ERROR: {e}")

# Forward pass with autocast (like actual E2E training)
print("\n--- Forward WITH autocast (bfloat16) ---")
try:
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
        output = model(input_ids, use_cache=False)
        logits = output['logits'] if isinstance(output, dict) else output[0]
    print(f"Logits shape: {logits.shape}")
    print(f"Logits: min={logits.min():.4f} max={logits.max():.4f} mean={logits.mean():.4f}")
    print(f"Has NaN: {logits.isnan().any().item()}")
    print(f"Has Inf: {logits.isinf().any().item()}")
except Exception as e:
    print(f"ERROR: {e}")

# Check intermediate hidden states layer by layer
print("\n--- Layer-by-layer hidden state check ---")
try:
    embed = model.model.embed_tokens(input_ids)
    print(f"Embed: shape={embed.shape} mean={embed.mean():.6f} std={embed.std():.6f} nan={embed.isnan().any().item()}")
    
    # Prepare position info
    seq_len = input_ids.shape[1]
    position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
    
    # Get causal mask
    causal_mask = model.model._update_causal_mask(
        None, embed, 
        torch.arange(seq_len, device=input_ids.device),
        None, False)
    
    # Get position embeddings
    position_embeddings = model.model.rotary_emb(embed, position_ids)
    
    hidden = embed
    for i, layer in enumerate(model.model.layers):
        hidden_out = layer(hidden, attention_mask=causal_mask, 
                          position_ids=position_ids,
                          position_embeddings=position_embeddings,
                          use_cache=False)[0]
        has_nan = hidden_out.isnan().any().item()
        has_inf = hidden_out.isinf().any().item()
        if has_nan or has_inf or i < 3 or i == len(model.model.layers) - 1:
            print(f"  Layer {i}: mean={hidden_out.float().mean():.6f} std={hidden_out.float().std():.6f} nan={has_nan} inf={has_inf}")
        if has_nan or has_inf:
            # Debug the problematic layer
            print(f"    >>> NaN/Inf detected at layer {i}! Checking sub-components...")
            # Check attention
            ln_out = layer.input_layernorm(hidden)
            print(f"    input_layernorm: nan={ln_out.isnan().any().item()} inf={ln_out.isinf().any().item()} mean={ln_out.float().mean():.6f} std={ln_out.float().std():.6f}")
            break
        hidden = hidden_out
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)
