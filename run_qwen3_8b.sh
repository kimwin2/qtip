#!/bin/bash
# ============================================================================
# Qwen3-8B QTIP Quantization Pipeline (Re-run after Hadamard fix)
# Step 3 (Quantize) → Step 4 (HFize) → Step 5 (E2E) → Step 6 (Eval PPL)
#
# Hessian은 재사용 (Hadamard 변환 미사용이므로 영향 없음)
# 기존 ckpt/hf 결과는 삭제 후 재생성
#
# Note: 8B (hidden=4096, intermediate=12288=12*1024)는 had12 사용으로
#       Hadamard 버그에 직접 영향받지 않지만, 안전을 위해 재실행
#
# 사용법: bash run_qwen3_8b.sh
# ============================================================================
set -e  # 에러 시 즉시 종료

# ===== HuggingFace Cache (pre-downloaded) =====
export HF_DATASETS_CACHE="/group-volume/ym1012.kim/repo/qtip/hf_cache/huggingface/datasets"
export HF_HUB_CACHE="/group-volume/ym1012.kim/repo/qtip/hf_cache/huggingface/hub"

# ===== Configuration =====
BASE_MODEL="/models/Qwen/Qwen3-8B-Base"
HESS_DIR="hessians/qwen3_8b"
CKPT_DIR="ckpt/qwen3_8b_2bit"
HF_DIR="hf/qwen3_8b_2bit"
HF_E2E_DIR="hf/qwen3_8b_2bit_e2e"
LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/qwen3_8b_2bit_rerun.log"

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
QUANT_BATCH_SIZE=4
QUANT_DEVSET_SIZE=256
QUANT_CTX_SIZE=4096

# ===== Create directories =====
mkdir -p ${LOG_DIR}

echo "==============================================" | tee -a ${LOG_FILE}
echo " Qwen3-8B QTIP Re-quantization (Hadamard fix)" | tee -a ${LOG_FILE}
echo "==============================================" | tee -a ${LOG_FILE}
echo "Model: ${BASE_MODEL}" | tee -a ${LOG_FILE}
echo "Quantization: ${K}-bit (L=${L}, K=${K}, V=${V})" | tee -a ${LOG_FILE}
echo "==============================================" | tee -a ${LOG_FILE}

# ============================================================================
# Clean old checkpoints
# ============================================================================
echo "" | tee -a ${LOG_FILE}
echo "===== Cleaning old checkpoints =====" | tee -a ${LOG_FILE}
rm -rf ${CKPT_DIR}
rm -rf ${HF_DIR}
rm -rf ${HF_E2E_DIR}
mkdir -p ${CKPT_DIR}
mkdir -p ${HF_DIR}
mkdir -p ${HF_E2E_DIR}
echo "Old ckpt/hf directories cleaned." | tee -a ${LOG_FILE}

# ============================================================================
# Step 3: QTIP Quantization
# ============================================================================
echo "" | tee -a ${LOG_FILE}
echo "===== Step 3: QTIP Quantization =====" | tee -a ${LOG_FILE}
echo "Running quantization (single-GPU)..." | tee -a ${LOG_FILE}
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

echo "Quantization complete!" | tee -a ${LOG_FILE}

# ============================================================================
# Step 4: Convert to HuggingFace model
# ============================================================================
echo "" | tee -a ${LOG_FILE}
echo "===== Step 4: HF Model Conversion =====" | tee -a ${LOG_FILE}
python -m quantize_llama.hfize_llama \
    --quantized_path ${CKPT_DIR} \
    --hf_output_path ${HF_DIR} \
    2>&1 | tee -a ${LOG_FILE}

echo "HF conversion complete!" | tee -a ${LOG_FILE}

# ============================================================================
# Step 5: End-to-End Finetuning
# ============================================================================
echo "" | tee -a ${LOG_FILE}
echo "===== Step 5: E2E Finetuning =====" | tee -a ${LOG_FILE}
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

echo "E2E finetuning complete!" | tee -a ${LOG_FILE}

# ============================================================================
# Step 6: Evaluate PPL (wikitext2 + c4)
# ============================================================================
echo "" | tee -a ${LOG_FILE}
echo "===== Step 6: PPL Evaluation =====" | tee -a ${LOG_FILE}

echo "--- Evaluating non-finetuned model ---" | tee -a ${LOG_FILE}
python -m eval.eval_ppl \
    --hf_path ${HF_DIR} \
    --base_model ${BASE_MODEL} \
    2>&1 | tee -a ${LOG_FILE}

echo "--- Evaluating E2E finetuned model ---" | tee -a ${LOG_FILE}
python -m eval.eval_ppl \
    --hf_path ${HF_E2E_DIR} \
    --base_model ${BASE_MODEL} \
    2>&1 | tee -a ${LOG_FILE}

echo "" | tee -a ${LOG_FILE}
echo "==============================================" | tee -a ${LOG_FILE}
echo " Pipeline Complete!" | tee -a ${LOG_FILE}
echo " Results are in: ${LOG_FILE}" | tee -a ${LOG_FILE}
echo "==============================================" | tee -a ${LOG_FILE}
