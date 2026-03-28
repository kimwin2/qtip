"""
Zero-shot evaluation using lm_eval 0.4.8
Evaluates quantized QTIP models on: HellaSwag, PIQA, WinoGrande, ARC-e, ARC-c

Usage:
    python -m eval.eval_zeroshot_v2 \
        --hf_path hf/qwen3_4b_2bit_e2e \
        --tasks hellaswag,piqa,winogrande,arc_easy,arc_challenge \
        --batch_size 16 \
        --output_path results/qwen3_4b_2bit_e2e_zeroshot.json
"""
import argparse
import json
import os
import random
import sys

import glog
import torch

from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from transformers import AutoTokenizer

from lib.linear import QuantizedLinear
from lib.utils.unsafe_import import model_from_hf_path

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

# Default 5 benchmarks
DEFAULT_TASKS = "hellaswag,piqa,winogrande,arc_easy,arc_challenge"

parser = argparse.ArgumentParser(description="Zero-shot evaluation with lm_eval 0.4.8")
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--hf_path', required=True, type=str,
                    help='Path to quantized HF model')
parser.add_argument('--batch_size', type=int, default=16,
                    help='Batch size for evaluation')
parser.add_argument("--tasks", type=str, default=DEFAULT_TASKS,
                    help=f"Comma-separated task names (default: {DEFAULT_TASKS})")
parser.add_argument("--output_path", default=None, type=str,
                    help="Path to save JSON results")
parser.add_argument('--num_fewshot', type=int, default=0,
                    help='Number of few-shot examples (0 = zero-shot)')
parser.add_argument('--limit', type=int, default=None,
                    help='Limit number of examples per task (for debugging)')
parser.add_argument('--manifest_model', action='store_true',
                    help='Manifest model weights for codebooks without kernel support')
parser.add_argument('--max_mem_ratio', type=float, default=0.7,
                    help='Maximum GPU memory ratio')


def format_results_table(results_dict):
    """Format results as a markdown-style table for easy reading."""
    lines = []
    lines.append("")
    lines.append("=" * 70)
    lines.append(f"{'Task':<20} {'Metric':<25} {'Value':>10}")
    lines.append("=" * 70)

    for task_name, metrics in results_dict.items():
        if task_name.startswith("_"):  # skip metadata keys
            continue
        first = True
        for metric_name, value in metrics.items():
            if metric_name in ('alias', 'group'):
                continue
            display_name = task_name if first else ""
            if isinstance(value, float):
                lines.append(f"{display_name:<20} {metric_name:<25} {value:>10.4f}")
            else:
                lines.append(f"{display_name:<20} {metric_name:<25} {str(value):>10}")
            first = False
        lines.append("-" * 70)

    lines.append("=" * 70)
    return "\n".join(lines)


def extract_accuracy_summary(results_dict):
    """Extract primary accuracy metrics for a clean summary."""
    summary = {}
    # Map task names to their primary accuracy metric
    acc_keys = ['acc_norm,none', 'acc,none', 'acc_norm', 'acc']

    for task_name, metrics in results_dict.items():
        if task_name.startswith("_"):
            continue
        for key in acc_keys:
            if key in metrics:
                summary[task_name] = metrics[key]
                break

    return summary


def main(args):
    glog.info(f"Loading model from: {args.hf_path}")
    model, model_str = model_from_hf_path(
        args.hf_path,
        max_mem_ratio=args.max_mem_ratio,
        device_map='balanced'
    )

    # Manifest for faster inference (non-kernel codebooks)
    if args.manifest_model:
        for module in model.modules():
            if isinstance(module, QuantizedLinear):
                module.mode = 'train-fixW'

    tokenizer = AutoTokenizer.from_pretrained(model_str)
    tokenizer.pad_token = tokenizer.eos_token
    glog.info("Model and tokenizer loaded!")

    task_names = [t.strip() for t in args.tasks.split(",")]
    glog.info(f"Tasks: {task_names}")

    # Wrap with lm_eval HFLM
    lm_eval_model = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
    )

    # Run evaluation
    glog.info("Starting evaluation...")
    results = evaluator.simple_evaluate(
        model=lm_eval_model,
        tasks=task_names,
        limit=args.limit,
        num_fewshot=args.num_fewshot,
    )

    # Print detailed results
    print(format_results_table(results['results']))

    # Print accuracy summary
    summary = extract_accuracy_summary(results['results'])
    print("\n===== Accuracy Summary =====")
    for task, acc in summary.items():
        print(f"  {task:<20}: {acc:.4f} ({acc*100:.2f}%)")

    avg_acc = sum(summary.values()) / len(summary) if summary else 0
    print(f"\n  {'Average':<20}: {avg_acc:.4f} ({avg_acc*100:.2f}%)")
    print("=" * 40)

    # Save results
    if args.output_path is not None:
        os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok=True)

        save_data = {
            'hf_path': args.hf_path,
            'tasks': task_names,
            'num_fewshot': args.num_fewshot,
            'batch_size': args.batch_size,
            'results': results['results'],
            'accuracy_summary': summary,
            'average_accuracy': avg_acc,
        }

        with open(args.output_path, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        glog.info(f"Results saved to: {args.output_path}")

    return results


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    main(args)
