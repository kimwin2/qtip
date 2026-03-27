"""
Diagnostic script for debugging quantized Qwen3 model PPL issues.
Run: python debug_ppl.py --hf_path hf/qwen3_1.7b_2bit --base_model /models/Qwen/Qwen3-1.7B-Base
"""
import argparse
import torch
import transformers
from lib.linear.quantized_linear import QuantizedLinear

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument('--hf_path', type=str, required=True)
parser.add_argument('--base_model', type=str, required=True)
args = parser.parse_args()

print("=" * 60)
print("1. CONFIG CHECK")
print("=" * 60)

config = transformers.AutoConfig.from_pretrained(args.hf_path)
print(f"model_type: {config.model_type}")
print(f"has quip_params: {hasattr(config, 'quip_params')}")
if hasattr(config, 'quip_params'):
    print(f"quip_params: {config.quip_params}")
print(f"tie_word_embeddings: {config.tie_word_embeddings}")
print(f"torch_dtype: {config.torch_dtype}")
print(f"hidden_act: {config.hidden_act}")
print(f"vocab_size: {config.vocab_size}")
print(f"hidden_size: {config.hidden_size}")
print(f"num_hidden_layers: {config.num_hidden_layers}")
print(f"num_attention_heads: {config.num_attention_heads}")
print(f"num_key_value_heads: {config.num_key_value_heads}")
print(f"_attn_implementation: {getattr(config, '_attn_implementation', 'NOT SET')}")
qk_norm = getattr(config, 'qk_norm', getattr(config, 'use_qk_norm', 'NOT SET'))
print(f"qk_norm: {qk_norm}")

print()
print("=" * 60)
print("2. LOADING QUANTIZED MODEL")
print("=" * 60)

from lib.utils.unsafe_import import model_from_hf_path
model, model_str = model_from_hf_path(args.hf_path)
print(f"model_str: {model_str}")
print(f"model class: {type(model).__name__}")

print()
print("=" * 60)
print("3. QUANTIZED LINEAR WEIGHT CHECK")
print("=" * 60)

# Check first layer's q_proj
layer0 = model.model.layers[0]
q_proj = layer0.self_attn.q_proj
print(f"q_proj type: {type(q_proj).__name__}")

if isinstance(q_proj, QuantizedLinear):
    print(f"  trellis shape: {q_proj.trellis.shape}, dtype: {q_proj.trellis.dtype}")
    print(f"  trellis all zeros: {(q_proj.trellis == 0).all().item()}")
    print(f"  trellis nonzero count: {q_proj.trellis.nonzero().shape[0]}")
    print(f"  SU shape: {q_proj.SU.shape}, dtype: {q_proj.SU.dtype}")
    print(f"  SU all ones: {(q_proj.SU == 1).all().item()}")
    print(f"  SU stats: min={q_proj.SU.min().item():.4f} max={q_proj.SU.max().item():.4f} mean={q_proj.SU.float().mean().item():.4f}")
    print(f"  SV shape: {q_proj.SV.shape}, dtype: {q_proj.SV.dtype}")
    print(f"  SV all ones: {(q_proj.SV == 1).all().item()}")
    print(f"  SV stats: min={q_proj.SV.min().item():.4f} max={q_proj.SV.max().item():.4f} mean={q_proj.SV.float().mean().item():.4f}")
    if q_proj.tlut is not None:
        print(f"  tlut shape: {q_proj.tlut.shape}, dtype: {q_proj.tlut.dtype}")
        print(f"  tlut all zeros: {(q_proj.tlut == 0).all().item()}")
        print(f"  tlut stats: min={q_proj.tlut.min().item():.4f} max={q_proj.tlut.max().item():.4f}")
    print(f"  rcp: {q_proj.rcp}")
    print(f"  decode_mode: {q_proj.decode_mode}")
    print(f"  has_kernel: {q_proj.has_kernel}")

print()
print("=" * 60)
print("4. EMBED/LM_HEAD WEIGHT CHECK")
print("=" * 60)

embed_w = model.model.embed_tokens.weight
lmhead_w = model.lm_head.weight
print(f"embed_tokens weight shape: {embed_w.shape}, dtype: {embed_w.dtype}")
print(f"embed_tokens stats: min={embed_w.min().item():.4f} max={embed_w.max().item():.4f} mean={embed_w.float().mean().item():.6f}")
print(f"embed_tokens all zeros: {(embed_w == 0).all().item()}")
print(f"lm_head weight shape: {lmhead_w.shape}, dtype: {lmhead_w.dtype}")
print(f"lm_head stats: min={lmhead_w.min().item():.4f} max={lmhead_w.max().item():.4f} mean={lmhead_w.float().mean().item():.6f}")
print(f"lm_head all zeros: {(lmhead_w == 0).all().item()}")
print(f"embed_tokens == lm_head (tied): {torch.equal(embed_w, lmhead_w)}")
print(f"embed_tokens is lm_head (same tensor): {embed_w.data_ptr() == lmhead_w.data_ptr()}")

# Compare with original model
print()
print("Loading base model embed_tokens for comparison...")
orig_model = transformers.AutoModelForCausalLM.from_pretrained(
    args.base_model, torch_dtype='auto', low_cpu_mem_usage=True)
orig_embed = orig_model.model.embed_tokens.weight
print(f"orig embed stats: min={orig_embed.min().item():.4f} max={orig_embed.max().item():.4f} mean={orig_embed.float().mean().item():.6f}")
print(f"embed_tokens match original: {torch.allclose(embed_w.cpu().float(), orig_embed.cpu().float(), atol=1e-3)}")
diff = (embed_w.cpu().float() - orig_embed.cpu().float()).abs()
print(f"embed_tokens diff: max={diff.max().item():.6f} mean={diff.mean().item():.6f}")

print()
print("Checking norm weights...")
norm_w = model.model.norm.weight
orig_norm = orig_model.model.norm.weight
print(f"norm stats: min={norm_w.min().item():.4f} max={norm_w.max().item():.4f}")
print(f"orig norm stats: min={orig_norm.min().item():.4f} max={orig_norm.max().item():.4f}")
print(f"norm match original: {torch.allclose(norm_w.cpu().float(), orig_norm.cpu().float(), atol=1e-3)}")

del orig_model
torch.cuda.empty_cache()

print()
print("=" * 60)
print("5. SINGLE FORWARD PASS CHECK")
print("=" * 60)

tokenizer = transformers.AutoTokenizer.from_pretrained(args.base_model)
test_text = "The capital of France is"
tokens = tokenizer(test_text, return_tensors='pt')
input_ids = tokens.input_ids.cuda()
print(f"Input: '{test_text}'")
print(f"Token IDs: {input_ids}")

output = model(input_ids, use_cache=False, output_hidden_states=True)
logits = output.logits if hasattr(output, 'logits') else output[0]
print(f"Logits shape: {logits.shape}")
print(f"Logits dtype: {logits.dtype}")
print(f"Logits stats: min={logits.min().item():.4f} max={logits.max().item():.4f} mean={logits.mean().item():.4f} std={logits.std().item():.4f}")

# Check if logits are meaningful (not random uniform)
probs = torch.softmax(logits[0, -1, :], dim=-1)
top5_probs, top5_ids = probs.topk(5)
print(f"\nTop 5 predictions for next token:")
for i in range(5):
    tok = tokenizer.decode([top5_ids[i].item()])
    print(f"  '{tok}' (id={top5_ids[i].item()}) prob={top5_probs[i].item():.4f}")

# Check entropy of last position - random would be ~log(vocab_size)
entropy = -(probs * probs.log().clamp(min=-100)).sum().item()
max_entropy = torch.log(torch.tensor(float(config.vocab_size))).item()
print(f"\nEntropy: {entropy:.2f} (random would be ~{max_entropy:.2f})")

# Check hidden states from intermediate layers
if hasattr(output, 'hidden_states') and output.hidden_states is not None:
    print(f"\nHidden states available: {len(output.hidden_states)} layers")
    for i in [0, len(output.hidden_states)//2, -1]:
        hs = output.hidden_states[i]
        print(f"  Layer {i}: shape={hs.shape} mean={hs.float().mean().item():.6f} std={hs.float().std().item():.6f}")

print()
print("=" * 60)
print("6. LAYER-BY-LAYER FORWARD CHECK")
print("=" * 60)

# Run input through embedding + first layer only, compare with original
embed_output = model.model.embed_tokens(input_ids)
print(f"Embedding output: shape={embed_output.shape} mean={embed_output.float().mean().item():.6f} std={embed_output.float().std().item():.6f}")

print()
print("=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)
