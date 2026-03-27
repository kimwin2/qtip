import argparse
import os
import time

import glog

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

import torch
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer

from lib import utils
from lib.algo import finetune
from lib.codebook import bitshift
from operator import attrgetter

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_cpu_threads', default=8, type=int)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--devset_size', default=384, type=int)
parser.add_argument('--ctx_size', default=4096, type=int)
parser.add_argument('--save_path', type=str)
parser.add_argument('--in_hess_path', type=str)
parser.add_argument('--base_model', type=str)
parser.add_argument('--sigma_reg', default=1e-2, type=float)
parser.add_argument('--sigma_reg2', default=1e-2, type=float)
parser.add_argument('--scale_override', default=-1, type=float)
parser.add_argument('--codebook', type=str)
parser.add_argument('--use_fp64', action='store_true')
parser.add_argument('--no_use_buffered', action='store_true')
parser.add_argument('--sample_proc', default=1, type=int)
parser.add_argument('--lowmem_ldlq', action='store_true')
parser.add_argument('--ft_lr', default=3e-6, type=float)
parser.add_argument('--ft_bs', default=4, type=int)
parser.add_argument('--ft_update_freq', default=1, type=int)
parser.add_argument('--ft_epochs', default=5, type=int)
parser.add_argument('--ft_valid_freq', default=1, type=int)
parser.add_argument('--ft_valid_size', default=128, type=float)
parser.add_argument('--ft_early_stop', default=5, type=int)
parser.add_argument('--ft_grad_ckpt', action='store_true')
parser.add_argument('--td_x', default=16, type=int)
parser.add_argument('--td_y', default=16, type=int)
parser.add_argument('--L', default=16, type=int)
parser.add_argument('--K', default=2, type=int)
parser.add_argument('--V', default=2, type=int)
parser.add_argument('--tlut_bits', default=0, type=int)
parser.add_argument('--decode_mode', default='lut', type=str)
parser.add_argument('--ft_train_lut', action='store_true')
parser.add_argument('--split_for_tp', action='store_true')
parser.add_argument('--tp_rank', default=8, type=int)
parser.add_argument('--skip_list', default=None, type=str)


def check_exist(idx, args):
    suffix = ['q', 'k', 'v', 'o', 'up', 'gate', 'down', 'layernorm']
    for _ in suffix:
        test = f'{args.save_path}/{idx}_{_}.pt'
        if not os.path.exists(test):
            return False
    return True


def quantize_decoder(layer, idx, cb, args, device, pre_orig_emb,
                     orig_emb, model_config, skip_list, layer_kwargs=None):
    if check_exist(idx, args):
        return

    if skip_list is None:
        skip_list = []

    # layer name, save_name, input hessian file, output hessian file
    quant_order = []
    for thing in [('self_attn.v_proj', 'v', 'qkv', 'v', 'col'),
                  ('self_attn.q_proj', 'q', 'qkv', 'q', 'col'),
                  ('self_attn.k_proj', 'k', 'qkv', 'k', 'col'),
                  ('self_attn.o_proj', 'o', 'o', 'o', 'row'),
                  ('mlp.up_proj', 'up', 'up', 'up', 'col'),
                  ('mlp.gate_proj', 'gate', 'up', 'gate', 'col'),
                  ('mlp.down_proj', 'down', 'down', 'down', 'row')]:
        if f'{idx}_{thing[1]}' not in skip_list:
            quant_order.append(thing)
        else:
            attrgetter(thing[0])(layer).weight.requires_grad = False
            print(f'skipping {idx}_{thing[1]}')

    finetune.quantize_finetune_decoder_layer(layer, quant_order, idx, cb, args,
                                             device, pre_orig_emb, orig_emb,
                                             layer_kwargs=layer_kwargs)

    # Save layernorm weights
    ln_dict = {
        'input_layernorm': layer.input_layernorm.weight,
        'post_attention_layernorm': layer.post_attention_layernorm.weight,
    }
    # Save q_norm/k_norm if present (Qwen3)
    if hasattr(layer.self_attn, 'q_norm'):
        ln_dict['q_norm'] = layer.self_attn.q_norm.weight
    if hasattr(layer.self_attn, 'k_norm'):
        ln_dict['k_norm'] = layer.self_attn.k_norm.weight

    torch.save(ln_dict, f'{args.save_path}/{idx}_layernorm.pt')


def main(args):
    if args.skip_list is not None:
        args.skip_list = args.skip_list.split(',')

    dtype_ = torch.float64 if args.use_fp64 else torch.float32

    cb = bitshift.bitshift_codebook(L=args.L,
                                    K=args.K,
                                    V=args.V,
                                    tlut_bits=args.tlut_bits,
                                    decode_mode=args.decode_mode)
    model = AutoModelForCausalLM.from_pretrained(args.base_model,
                                                 torch_dtype='auto',
                                                 low_cpu_mem_usage=True)

    # save configs
    all_config = {'quant_args': args, 'model_config': model.config}
    quip_params = {
        'codebook': args.codebook,
        'codebook_version': cb.version,
        'L': args.L,
        'K': args.K,
        'V': args.V,
        'tlut_bits': args.tlut_bits,
        'decode_mode': args.decode_mode,
        'td_x': args.td_x,
        'td_y': args.td_y,
        'split_for_tp': args.split_for_tp,
        'skip_list': args.skip_list,
    }
    all_config['model_config'].update({'quip_params': quip_params})
    torch.save(all_config, os.path.join(args.save_path, 'config.pt'))

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    glog.info('loaded model')

    devset = utils.sample_rp1t(tokenizer, args.devset_size, args.ctx_size,
                               args.sample_proc)
    glog.info('loaded dataset and devset')

    # === Single-GPU Sequential Processing ===
    device = torch.device('cuda:0')

    # Compute initial embeddings
    orig_emb = model.model.embed_tokens(devset)
    glog.info(f'computed initial embeddings, shape: {orig_emb.shape}')

    # Capture kwargs needed for decoder layer forward pass
    # Use a Catcher to get position_ids, attention_mask, etc.
    class KwargsCatcher(torch.nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.layer = layer
            self.captured_args = None
            self.captured_kwargs = None

        def forward(self, *args, **kwargs):
            self.captured_args = args
            self.captured_kwargs = kwargs
            raise StopIteration

    # Get the kwargs by running a dummy forward through the full model
    catcher = KwargsCatcher(model.model.layers[0])
    original_layer = model.model.layers[0]
    model.model.layers[0] = catcher
    try:
        dummy_input = devset[:args.batch_size].to(device)
        model.to(device)
        model(dummy_input, use_cache=False)
    except StopIteration:
        pass
    model.model.layers[0] = original_layer
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
                    layer_kwargs[key] = tuple(v.cpu() if isinstance(v, torch.Tensor) else v for v in val)
                else:
                    layer_kwargs[key] = val
    del catcher
    utils.clean()

    # Re-compute embeddings on CPU
    orig_emb = model.model.embed_tokens(devset)

    # Process each layer sequentially on single GPU
    for i in range(len(model.model.layers)):
        glog.info(f'=== Processing layer {i} ===')
        st = time.time()

        # Move layer to GPU
        model.model.layers[i].to(device)

        # Compute output embeddings for this layer
        new_emb = torch.zeros_like(orig_emb)
        for j in range(0, args.devset_size, args.batch_size):
            end_j = min(j + args.batch_size, args.devset_size)
            batch_input = orig_emb[j:end_j].to(device)

            # Prepare kwargs for this batch
            fwd_kwargs = {
                'use_cache': False,
                'output_attentions': False,
            }
            for key, val in layer_kwargs.items():
                if isinstance(val, torch.Tensor):
                    fwd_kwargs[key] = val.to(device)
                elif isinstance(val, tuple):
                    fwd_kwargs[key] = tuple(v.to(device) if isinstance(v, torch.Tensor) else v for v in val)
                else:
                    fwd_kwargs[key] = val

            with torch.no_grad():
                output = model.model.layers[i](batch_input, **fwd_kwargs)[0]
            new_emb[j:end_j] = output.cpu()

            del batch_input, output
            utils.clean()

        model.model.layers[i].cpu()
        utils.clean()
        glog.info(f'computed original embedding for layer {i} in {time.time() - st:.1f}s')

        # Quantize this layer (single GPU, synchronous)
        quantize_decoder(
            model.model.layers[i],
            i,
            cb,
            args,
            device,
            orig_emb,      # pre-layer embeddings
            new_emb,        # post-layer embeddings
            all_config['model_config'],
            args.skip_list,
            layer_kwargs=layer_kwargs
        )

        # Update embeddings for next layer
        orig_emb = new_emb

        # Free layer memory
        model.model.layers[i] = None
        utils.clean()
        glog.info(f'=== Done layer {i} ===')


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.save_path, exist_ok=True)
    main(args)
