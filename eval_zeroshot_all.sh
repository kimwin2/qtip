#!/bin/bash
# ============================================================================
# Zero-shot Benchmark Evaluation for Qwen3 1.7B / 4B / 8B (2-bit QTIP)
#
# Benchmarks: HellaSwag, PIQA, WinoGrande, ARC-Easy, ARC-Challenge
# Framework:  lm_eval 0.4.8
#
# 사용법: bash eval_zeroshot_all.sh
#         bash eval_zeroshot_all.sh --limit 100  # 디버깅용 (샘플 수 제한)
# ============================================================================
set -e

# ===== HuggingFace Cache (pre-downloaded) =====
export HF_DATASETS_CACHE="/group-volume/ym1012.kim/repo/qtip/hf_cache/huggingface/datasets"
export HF_HUB_CACHE="/group-volume/ym1012.kim/repo/qtip/hf_cache/huggingface/hub"

# ===== Configuration =====
TASKS="hellaswag,piqa,winogrande,arc_easy,arc_challenge"
BATCH_SIZE=16
NUM_FEWSHOT=0
RESULT_DIR="results/zeroshot"
LOG_DIR="logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Parse optional arguments
LIMIT_ARG=""
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --limit) LIMIT_ARG="--limit $2"; shift ;;
        --batch_size) BATCH_SIZE=$2; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
    shift
done

# Models to evaluate (E2E finetuned versions)
declare -A MODELS
MODELS["qwen3_1.7b_2bit"]="hf/qwen3_1.7b_2bit_e2e"
MODELS["qwen3_4b_2bit"]="hf/qwen3_4b_2bit_e2e"
MODELS["qwen3_8b_2bit"]="hf/qwen3_8b_2bit_e2e"

# ===== Create directories =====
mkdir -p ${RESULT_DIR}
mkdir -p ${LOG_DIR}

echo "============================================================"
echo " Zero-shot Benchmark Evaluation"
echo "============================================================"
echo " Tasks:     ${TASKS}"
echo " Fewshot:   ${NUM_FEWSHOT}"
echo " Batch:     ${BATCH_SIZE}"
echo " Results:   ${RESULT_DIR}/"
echo " Limit:     ${LIMIT_ARG:-none (full eval)}"
echo "============================================================"
echo ""

# ===== Run evaluations =====
for MODEL_NAME in "qwen3_1.7b_2bit" "qwen3_4b_2bit" "qwen3_8b_2bit"; do
    HF_PATH="${MODELS[$MODEL_NAME]}"
    OUTPUT_FILE="${RESULT_DIR}/${MODEL_NAME}_zeroshot_${TIMESTAMP}.json"
    LOG_FILE="${LOG_DIR}/${MODEL_NAME}_zeroshot_${TIMESTAMP}.log"

    echo "============================================================"
    echo " Evaluating: ${MODEL_NAME}"
    echo " Model path: ${HF_PATH}"
    echo " Output:     ${OUTPUT_FILE}"
    echo "============================================================"

    if [ ! -d "${HF_PATH}" ]; then
        echo "WARNING: Model directory not found: ${HF_PATH}"
        echo "Skipping ${MODEL_NAME}..."
        echo ""
        continue
    fi

    python -m eval.eval_zeroshot_v2 \
        --hf_path ${HF_PATH} \
        --tasks ${TASKS} \
        --batch_size ${BATCH_SIZE} \
        --num_fewshot ${NUM_FEWSHOT} \
        --output_path ${OUTPUT_FILE} \
        ${LIMIT_ARG} \
        2>&1 | tee ${LOG_FILE}

    echo ""
    echo "${MODEL_NAME} evaluation complete!"
    echo "Results saved to: ${OUTPUT_FILE}"
    echo ""
done

# ===== Print combined summary =====
echo ""
echo "============================================================"
echo " All Evaluations Complete!"
echo "============================================================"
echo " Results directory: ${RESULT_DIR}/"
echo ""

# Print a summary table from all result files
echo "===== Combined Results Summary ====="
echo ""
python3 -c "
import json, glob, os, sys

result_files = sorted(glob.glob('${RESULT_DIR}/*_zeroshot_${TIMESTAMP}.json'))
if not result_files:
    print('No result files found.')
    sys.exit(0)

# Collect all results
all_results = {}
all_tasks = set()
for f in result_files:
    with open(f) as fh:
        data = json.load(fh)
    name = os.path.basename(f).replace('_zeroshot_${TIMESTAMP}.json', '')
    all_results[name] = data.get('accuracy_summary', {})
    all_tasks.update(all_results[name].keys())

tasks = sorted(all_tasks)
models = sorted(all_results.keys())

# Print table header
header = f\"{'Model':<25}\" + ''.join(f'{t:<15}' for t in tasks) + f\"{'Avg':>8}\"
print(header)
print('=' * len(header))

for model in models:
    accs = all_results[model]
    vals = [accs.get(t, 0) for t in tasks]
    avg = sum(vals) / len(vals) if vals else 0
    row = f'{model:<25}' + ''.join(f'{v*100:<15.2f}' for v in vals) + f'{avg*100:>8.2f}'
    print(row)

print('=' * len(header))
" 2>/dev/null || echo "(Summary generation requires completed results)"

echo ""
echo "Done!"
