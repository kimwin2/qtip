"""
Single-GPU Hessian extraction for QTIP quantization.
Rewritten from the original multi-GPU multiprocessing version.
Supports both Llama and Qwen3 architectures.
"""
import argparse
import datetime
import os
import random
from copy import deepcopy

from tqdm import tqdm

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

import numpy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lib import utils

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--batch_size', default=2, type=int)
parser.add_argument('--devset_size', default=256, type=int)
parser.add_argument('--ctx_size', default=4096, type=int)
parser.add_argument('--base_model',
                    default='meta-llama/Llama-2-70b-hf',
                    type=str)
parser.add_argument('--save_path', default='hessians/llama2_70b', type=str)
parser.add_argument('--sample_proc', default=4, type=int)


def main(args):
    device = torch.device('cuda:0')

    print("loading model...")
    model = AutoModelForCausalLM.from_pretrained(args.base_model,
                                                 torch_dtype="auto",
                                                 low_cpu_mem_usage=True)
    print("loaded model!")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    print("loading dataset...")
    devset = utils.sample_rp1t(tokenizer,
                               args.devset_size,
                               args.ctx_size,
                               nproc=args.sample_proc)
    dev_emb = model.model.embed_tokens(devset)
    print("loaded dataset!")
    print(f"dev_emb dtype: {dev_emb.dtype}, shape: {dev_emb.shape}")

    # Capture kwargs needed for decoder layer forward
    class KwargsCatcher(torch.nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.layer = layer
            self.captured_kwargs = None

        def forward(self, *args, **kwargs):
            self.captured_kwargs = kwargs
            raise StopIteration

    # Get the kwargs by running a tiny forward through the model
    catcher = KwargsCatcher(model.model.layers[0])
    original_layer_0 = model.model.layers[0]
    model.model.layers[0] = catcher
    try:
        dummy_input = devset[:args.batch_size].to(device)
        model.to(device)
        model(dummy_input, use_cache=False)
    except StopIteration:
        pass
    model.model.layers[0] = original_layer_0
    model.cpu()
    utils.clean()

    # Extract captured kwargs
    layer_kwargs = {}
    if catcher.captured_kwargs is not None:
        for key in ['position_ids', 'attention_mask', 'position_embeddings']:
            if key in catcher.captured_kwargs and catcher.captured_kwargs[key] is not None:
                val = catcher.captured_kwargs[key]
                if isinstance(val, torch.Tensor):
                    layer_kwargs[key] = val.cpu()
                elif isinstance(val, tuple):
                    layer_kwargs[key] = tuple(
                        v.cpu() if isinstance(v, torch.Tensor) else v for v in val)
                else:
                    layer_kwargs[key] = val
    del catcher
    utils.clean()

    # Re-compute embeddings since model was moved
    dev_emb = model.model.embed_tokens(devset)

    # Process each layer sequentially on single GPU
    for transformer_layer_index in range(len(model.model.layers)):
        save_pfx = f'{args.save_path}/{transformer_layer_index}'

        # Check if already done
        all_done = all(
            os.path.exists(f'{save_pfx}_{key}.pt')
            for key in ['qkv', 'o', 'up', 'down']
        )
        if all_done:
            print(f"layer {transformer_layer_index} already done, computing forward pass only")
            # Still need to compute forward pass for correct activations
            layer = model.model.layers[transformer_layer_index]
            layer.to(device)
            for di in range(0, len(dev_emb), args.batch_size):
                end_di = min(di + args.batch_size, len(dev_emb))
                batch_input = dev_emb[di:end_di].to(device)
                fwd_kwargs = {'use_cache': False, 'output_attentions': False}
                for key, val in layer_kwargs.items():
                    if isinstance(val, torch.Tensor):
                        fwd_kwargs[key] = val.to(device)
                    elif isinstance(val, tuple):
                        fwd_kwargs[key] = tuple(
                            v.to(device) if isinstance(v, torch.Tensor) else v for v in val)
                    else:
                        fwd_kwargs[key] = val
                with torch.no_grad():
                    dev_emb[di:end_di] = layer(batch_input, **fwd_kwargs)[0].cpu()
                del batch_input
                utils.clean()
            layer.cpu()
            utils.clean()
            print(f"done processing layer {transformer_layer_index} (skipped hessian)")
            continue

        print(f"processing layer {transformer_layer_index}")

        layer = model.model.layers[transformer_layer_index]
        layer.to(device)

        # Register Hessian hooks
        done_qkv = utils.register_H_hook(layer.self_attn.q_proj, device)
        done_o = utils.register_H_hook(layer.self_attn.o_proj, device)
        done_up = utils.register_H_hook(layer.mlp.up_proj, device)
        done_down = utils.register_H_hook(layer.mlp.down_proj, device)

        # Forward pass through the layer
        for di in range(0, len(dev_emb), args.batch_size):
            end_di = min(di + args.batch_size, len(dev_emb))
            batch_input = dev_emb[di:end_di].to(device)

            # Prepare kwargs
            fwd_kwargs = {'use_cache': False, 'output_attentions': False}
            for key, val in layer_kwargs.items():
                if isinstance(val, torch.Tensor):
                    fwd_kwargs[key] = val.to(device)
                elif isinstance(val, tuple):
                    fwd_kwargs[key] = tuple(
                        v.to(device) if isinstance(v, torch.Tensor) else v for v in val)
                else:
                    fwd_kwargs[key] = val

            with torch.no_grad():
                output = layer(batch_input, **fwd_kwargs)[0].cpu()
            dev_emb[di:end_di] = output

            del batch_input, output
            utils.clean()

        layer.cpu()
        utils.clean()

        # Collect and save Hessians
        fn_dict = {
            'qkv': done_qkv,
            'o': done_o,
            'up': done_up,
            'down': done_down
        }
        for key in fn_dict:
            H, mu, ct = fn_dict[key]()
            mu.div_(ct)
            H.div_(ct)
            H.addmm_(-mu.unsqueeze(-1), mu.unsqueeze(0))

            save_path = f"{save_pfx}_{key}.pt"
            torch.save({
                'flatH': utils.sym_to_flat(H.to(torch.float32)),
                'mu': mu.to(torch.float32),
                'n': H.shape[0],
                'ct': ct
            }, save_path)
            print(f"  saved {save_path}")

            del H, mu
            utils.clean()

        # Free the layer
        model.model.layers[transformer_layer_index] = None
        utils.clean()

        print(f"done processing layer {transformer_layer_index}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    numpy.random.seed(args.seed)
    os.makedirs(args.save_path, exist_ok=True)
    main(args)
