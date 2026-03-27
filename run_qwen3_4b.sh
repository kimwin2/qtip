#!/bin/bash
# ============================================================================
# Qwen3-4B QTIP Quantization Pipeline
# 전체 파이프라인: Hessian → Quantize → HFize → E2E Finetune → Eval PPL
#
# 사용법: bash run_qwen3_4b.sh
# ============================================================================
set -e  # 에러 시 즉시 종료

# ===== HuggingFace Cache (pre-downloaded) =====
export HF_DATASETS_CACHE="/group-volume/ym1012.kim/repo/qtip/hf_cache/huggingface/datasets"
export HF_HUB_CACHE="/group-volume/ym1012.kim/repo/qtip/hf_cache/huggingface/hub"

# ===== Configuration =====
BASE_MODEL="/models/Qwen/Qwen3-4B-Base"
HESS_DIR="hessians/qwen3_4b"
CKPT_DIR="ckpt/qwen3_4b_2bit"
HF_DIR="hf/qwen3_4b_2bit"
HF_E2E_DIR="hf/qwen3_4b_2bit_e2e"
LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/qwen3_4b_2bit.log"

# Quantization parameters (2-bit HYB)
L=16
K=2
V=2
TD_X=16
TD_Y=16
DECODE_MODE="quantlut_sym"
TLUT_BITS=9
SCALE=0.9

# Training parameters
HESS_BATCH_SIZE=2
HESS_DEVSET_SIZE=256
HESS_CTX_SIZE=4096
QUANT_BATCH_SIZE=8
QUANT_DEVSET_SIZE=256
QUANT_CTX_SIZE=4096

# Transformers: use 4.51.3+ throughout (Qwen3 support required)

# ===== Create directories =====
mkdir -p ${HESS_DIR}
mkdir -p ${CKPT_DIR}
mkdir -p ${HF_DIR}
mkdir -p ${HF_E2E_DIR}
mkdir -p ${LOG_DIR}

echo "=============================================="
echo " Qwen3-4B QTIP Quantization Pipeline"
echo "=============================================="
echo "Model: ${BASE_MODEL}"
echo "Quantization: ${K}-bit (L=${L}, K=${K}, V=${V})"
echo "Log: ${LOG_FILE}"
echo "=============================================="

# ============================================================================
# Step 1: Hessian Extraction (quip-sharp)
# ============================================================================
echo ""
echo "===== Step 1: Hessian Extraction ====="

cd quip-sharp

echo "Running hessian extraction (single-GPU)..."
python -m quantize_llama.hessian_offline_llama \
    --base_model ${BASE_MODEL} \
    --save_path ../${HESS_DIR} \
    --batch_size ${HESS_BATCH_SIZE} \
    --devset_size ${HESS_DEVSET_SIZE} \
    --ctx_size ${HESS_CTX_SIZE} \
    --sample_proc 1 \
    2>&1 | tee -a ../${LOG_FILE}

cd ..
echo "Hessian extraction complete!"

# ============================================================================
# Step 3: QTIP Quantization
# ============================================================================
echo ""
echo "===== Step 3: QTIP Quantization ====="
echo "Running quantization (single-GPU)..."
python -m quantize_llama.quantize_finetune_llama \
    --save_path ${CKPT_DIR} \
    --codebook bitshift \
    --base_model ${BASE_MODEL} \
    --in_hess_path ${HESS_DIR} \
    --scale_override ${SCALE} \
    --ft_epochs 5 \
    --td_x ${TD_X} \
    --td_y ${TD_Y} \
    --L ${L} \
    --K ${K} \
    --V ${V} \
    --decode_mode ${DECODE_MODE} \
    --tlut_bits ${TLUT_BITS} \
    --batch_size ${QUANT_BATCH_SIZE} \
    --devset_size ${QUANT_DEVSET_SIZE} \
    --ctx_size ${QUANT_CTX_SIZE} \
    2>&1 | tee -a ${LOG_FILE}

echo "Quantization complete!"

# ============================================================================
# Step 4: Convert to HuggingFace model
# ============================================================================
echo ""
echo "===== Step 4: HF Model Conversion ====="
python -m quantize_llama.hfize_llama \
    --quantized_path ${CKPT_DIR} \
    --hf_output_path ${HF_DIR} \
    2>&1 | tee -a ${LOG_FILE}

echo "HF conversion complete!"

# ============================================================================
# Step 5: End-to-End Finetuning (requires 2 GPUs)
# ============================================================================
echo ""
echo "===== Step 5: E2E Finetuning (2 GPUs) ====="
python -m quantize_llama.finetune_e2e_llama \
    --base_model ${BASE_MODEL} \
    --hf_path ${HF_DIR} \
    --devset_size 256 \
    --ft_valid_size 64 \
    --ft_epochs 4 \
    --ft_update_freq 2 \
    --ft_bs 1 \
    --ctx_size 2048 \
    --ft_train_lut \
    --hf_output_path ${HF_E2E_DIR} \
    2>&1 | tee -a ${LOG_FILE}

echo "E2E finetuning complete!"

# ============================================================================
# Step 6: Evaluate PPL (wikitext2 + c4)
# ============================================================================
echo ""
echo "===== Step 6: PPL Evaluation ====="

echo "--- Evaluating non-finetuned model ---"
python -m eval.eval_ppl \
    --hf_path ${HF_DIR} \
    --base_model ${BASE_MODEL} \
    2>&1 | tee -a ${LOG_FILE}

echo "--- Evaluating E2E finetuned model ---"
python -m eval.eval_ppl \
    --hf_path ${HF_E2E_DIR} \
    --base_model ${BASE_MODEL} \
    2>&1 | tee -a ${LOG_FILE}

echo ""
echo "=============================================="
echo " Pipeline Complete!"
echo " Results are in: ${LOG_FILE}"
echo "=============================================="
