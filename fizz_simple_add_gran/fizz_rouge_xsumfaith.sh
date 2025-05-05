#!/bin/bash

# 기본 경로 및 값 설정 (필요에 따라 수정 가능)
INPUT_PATH="data/xsumfaith.json"
OUTPUT_PATH="results_addgran/fizz_rouge_xsumfaith.json"
DOC_LABEL="document"
SUMMARY_LABEL="claim"
LABEL_LABEL="label"
SCORE_COLUMN="FIZZ_score"
MODEL_NAME="orca2"
GRANULARITY="4G"
CUDA_DEVICE="0"
MODE="rouge_sbert"
WEIGHT_ROUGE="1"
WEIGHT_BERT="0"

# 인자 처리 (입력값이 있으면 덮어씌우기)
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --input_path) INPUT_PATH="$2"; shift ;;
        --output_path) OUTPUT_PATH="$2"; shift ;;
        --doc_label) DOC_LABEL="$2"; shift ;;
        --summary_label) SUMMARY_LABEL="$2"; shift ;;
        --label_label) LABEL_LABEL="$2"; shift ;;
        --score_column) SCORE_COLUMN="$2"; shift ;;
        --model_name) MODEL_NAME="$2"; shift ;;
        --granularity) GRANULARITY="$2"; shift ;;
        --cuda_device) CUDA_DEVICE="$2"; shift ;;
        --mode) MODE="$2"; shift ;;
        --weight_rouge) WEIGHT_ROUGE="$2"; shift ;;
        --weight_bert) WEIGHT_BERT="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# 파이썬 스크립트 실행
python fizz/fizz_rouge_xsumfaith.py \
    --input_path "$INPUT_PATH" \
    --output_path "$OUTPUT_PATH" \
    --doc_label "$DOC_LABEL" \
    --summary_label "$SUMMARY_LABEL" \
    --label_label "$LABEL_LABEL" \
    --score_column "$SCORE_COLUMN" \
    --model_name "$MODEL_NAME" \
    --granularity "$GRANULARITY" \
    --cuda_device "$CUDA_DEVICE" \
    --mode "$MODE" \
    --weight_rouge "$WEIGHT_ROUGE" \
    --weight_bert "$WEIGHT_BERT"