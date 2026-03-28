#!/bin/bash
# ============================================================================
# Zero-shot Benchmark Evaluation: Qwen3-8B (2-bit QTIP)
# Benchmarks: HellaSwag, PIQA, WinoGrande, ARC-Easy, ARC-Challenge
# Framework:  lm_eval 0.4.8
#
# 사용법: bash eval_zeroshot_8b.sh
#         bash eval_zeroshot_8b.sh --limit 100  # 디버깅용
# ============================================================================
set -e

# ===== HuggingFace Cache =====
export HF_DATASETS_CACHE="/group-volume/ym1012.kim/repo/qtip/hf_cache/huggingface/datasets"
export HF_HUB_CACHE="/group-volume/ym1012.kim/repo/qtip/hf_cache/huggingface/hub"

# ===== Configuration =====
MODEL_NAME="qwen3_8b_2bit"
HF_PATH="hf/qwen3_8b_2bit_e2e"
TASKS="hellaswag,piqa,winogrande,arc_easy,arc_challenge"
BATCH_SIZE=8
NUM_FEWSHOT=0
RESULT_DIR="results/zeroshot"
LOG_DIR="logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="${RESULT_DIR}/${MODEL_NAME}_zeroshot_${TIMESTAMP}.json"
LOG_FILE="${LOG_DIR}/${MODEL_NAME}_zeroshot_${TIMESTAMP}.log"

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

mkdir -p ${RESULT_DIR}
mkdir -p ${LOG_DIR}

echo "============================================================"
echo " Zero-shot Evaluation: Qwen3-8B (2-bit QTIP)"
echo "============================================================"
echo " Model:   ${HF_PATH}"
echo " Tasks:   ${TASKS}"
echo " Batch:   ${BATCH_SIZE}"
echo " Output:  ${OUTPUT_FILE}"
echo " Limit:   ${LIMIT_ARG:-none (full eval)}"
echo "============================================================"

python -m eval.eval_zeroshot_v2 \
    --hf_path ${HF_PATH} \
    --tasks ${TASKS} \
    --batch_size ${BATCH_SIZE} \
    --num_fewshot ${NUM_FEWSHOT} \
    --output_path ${OUTPUT_FILE} \
    ${LIMIT_ARG} \
    2>&1 | tee ${LOG_FILE}

echo ""
echo "============================================================"
echo " Evaluation Complete: ${MODEL_NAME}"
echo " Results: ${OUTPUT_FILE}"
echo "============================================================"
