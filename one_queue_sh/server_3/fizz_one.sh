#!/bin/bash

# 로그 디렉토리와 파일 설정
LOG_DIR="./log"
LOG_FILE="$LOG_DIR/gpu1.log"

# 로그 디렉토리 생성
mkdir -p $LOG_DIR

# 로그 파일 초기화
echo "===== 실행 시간 기록 시작 =====" > $LOG_FILE
echo "실행 시작 시간: $(date)" >> $LOG_FILE
echo "------------------------------------" >> $LOG_FILE

# 스크립트와 인자 목록
scripts=(

# "python fizz_coarse/fizz_coarse.py \
# --input_path data/data_aggrefact/aggre_fact_cnndm_sota.json \
# --output_path results_coarse/cnndm_tune/coarse_wr0p3_wb0p7_wc1_wm0p7_ww0p3.json \
# --doc_label document --summary_label claim --label_label label --score_column FIZZ_score --model_name orca2 \
# --granularity 3G \
# --cuda_device 1 \
# --weight_rouge 0.3 \
# --weight_bert 0.7 \
# --weight_contradiction 1 \
# --weight_min 0.7 \
# --weight_mean 0.3"

# "python fizz_coarse/fizz_coarse.py \
# --input_path data/data_aggrefact/aggre_fact_cnndm_sota.json \
# --output_path results_coarse/cnndm_tune/coarse_wr0p3_wb0p7_wc1_wm1_ww0.json \
# --doc_label document --summary_label claim --label_label label --score_column FIZZ_score --model_name orca2 \
# --granularity 3G \
# --cuda_device 1 \
# --weight_rouge 0.3 \
# --weight_bert 0.7 \
# --weight_contradiction 1 \
# --weight_min 1.0 \
# --weight_mean 0.0"

# "python fizz_coarse/fizz_coarse.py \
# --input_path data/data_aggrefact/aggre_fact_cnndm_sota.json \
# --output_path results_coarse/cnndm_tune/coarse_wr0p5_wb0p5_wc1_wm0_ww1.json \
# --doc_label document --summary_label claim --label_label label --score_column FIZZ_score --model_name orca2 \
# --granularity 3G \
# --cuda_device 1 \
# --weight_rouge 0.5 \
# --weight_bert 0.5 \
# --weight_contradiction 1 \
# --weight_min 0.0 \
# --weight_mean 1.0"

# "python fizz_coarse/fizz_coarse.py \
# --input_path data/data_aggrefact/aggre_fact_cnndm_sota.json \
# --output_path results_coarse/cnndm_tune/coarse_wr0p5_wb0p5_wc1_wm0p3_ww0p7.json \
# --doc_label document --summary_label claim --label_label label --score_column FIZZ_score --model_name orca2 \
# --granularity 3G \
# --cuda_device 1 \
# --weight_rouge 0.5 \
# --weight_bert 0.5 \
# --weight_contradiction 1 \
# --weight_min 0.3 \
# --weight_mean 0.7"

# "python fizz_coarse/fizz_coarse.py \
# --input_path data/data_aggrefact/aggre_fact_cnndm_sota.json \
# --output_path results_coarse/cnndm_tune/coarse_wr0p5_wb0p5_wc1_wm0p5_ww0p5.json \
# --doc_label document --summary_label claim --label_label label --score_column FIZZ_score --model_name orca2 \
# --granularity 3G \
# --cuda_device 1 \
# --weight_rouge 0.5 \
# --weight_bert 0.5 \
# --weight_contradiction 1 \
# --weight_min 0.5 \
# --weight_mean 0.5"

# "python fizz_coarse/fizz_coarse.py \
# --input_path data/data_aggrefact/aggre_fact_xsum_sota.json \
# --output_path results_coarse/xsum_tune/coarse_wr0p3_wb0p7_wc1_wm0p7_ww0p3.json \
# --doc_label document --summary_label claim --label_label label --score_column FIZZ_score --model_name orca2 \
# --granularity 3G \
# --cuda_device 1 \
# --weight_rouge 0.3 \
# --weight_bert 0.7 \
# --weight_contradiction 1 \
# --weight_min 0.7 \
# --weight_mean 0.3"

# "python fizz_coarse/fizz_coarse.py \
# --input_path data/data_aggrefact/aggre_fact_xsum_sota.json \
# --output_path results_coarse/xsum_tune/coarse_wr0p3_wb0p7_wc1_wm1_ww0.json \
# --doc_label document --summary_label claim --label_label label --score_column FIZZ_score --model_name orca2 \
# --granularity 3G \
# --cuda_device 1 \
# --weight_rouge 0.3 \
# --weight_bert 0.7 \
# --weight_contradiction 1 \
# --weight_min 1.0 \
# --weight_mean 0.0"

# "python fizz_coarse/fizz_coarse.py \
# --input_path data/data_aggrefact/aggre_fact_xsum_sota.json \
# --output_path results_coarse/xsum_tune/coarse_wr0p5_wb0p5_wc1_wm0_ww1.json \
# --doc_label document --summary_label claim --label_label label --score_column FIZZ_score --model_name orca2 \
# --granularity 3G \
# --cuda_device 1 \
# --weight_rouge 0.5 \
# --weight_bert 0.5 \
# --weight_contradiction 1 \
# --weight_min 0.0 \
# --weight_mean 1.0"

# "python fizz_coarse/fizz_coarse.py \
# --input_path data/data_aggrefact/aggre_fact_xsum_sota.json \
# --output_path results_coarse/xsum_tune/coarse_wr0p5_wb0p5_wc1_wm0p3_ww0p7.json \
# --doc_label document --summary_label claim --label_label label --score_column FIZZ_score --model_name orca2 \
# --granularity 3G \
# --cuda_device 1 \
# --weight_rouge 0.5 \
# --weight_bert 0.5 \
# --weight_contradiction 1 \
# --weight_min 0.3 \
# --weight_mean 0.7"

# "python fizz_coarse/fizz_coarse.py \
# --input_path data/data_aggrefact/aggre_fact_xsum_sota.json \
# --output_path results_coarse/xsum_tune/coarse_wr0p5_wb0p5_wc1_wm0p5_ww0p5.json \
# --doc_label document --summary_label claim --label_label label --score_column FIZZ_score --model_name orca2 \
# --granularity 3G \
# --cuda_device 1 \
# --weight_rouge 0.5 \
# --weight_bert 0.5 \
# --weight_contradiction 1 \
# --weight_min 0.5 \
# --weight_mean 0.5"


# "python fizz_coarse/fizz_coarse.py \
# --input_path data/data_aggrefact/aggre_fact_xsum_sota.json \
# --output_path results_coarse/xsum_tune/coarse_wr0_wb1_wc1_wm0p5_ww0p5.json \
# --doc_label document --summary_label claim --label_label label --score_column FIZZ_score --model_name orca2 \
# --granularity 3G \
# --cuda_device 1 \
# --weight_rouge 0.0 \
# --weight_bert 1.0 \
# --weight_contradiction 1 \
# --weight_min 0.5 \
# --weight_mean 0.5"

# "python fizz_coarse/fizz_coarse.py \
# --input_path data/data_aggrefact/aggre_fact_xsum_sota.json \
# --output_path results_coarse/xsum_tune/coarse_wr0_wb1_wc1_wm1_ww0.json \
# --doc_label document --summary_label claim --label_label label --score_column FIZZ_score --model_name orca2 \
# --granularity 3G \
# --cuda_device 6 \
# --weight_rouge 0.0 \
# --weight_bert 1.0 \
# --weight_contradiction 1 \
# --weight_min 1.0 \ 
# --weight_mean 0.0"

# "python fizz_coarse/fizz_coarse.py \
# --input_path data/data_aggrefact/aggre_fact_xsum_sota.json \
# --output_path results_coarse/xsum_tune/coarse_wr0p3_wb0p7_wc1_wm0_ww1.json \
# --doc_label document --summary_label claim --label_label label --score_column FIZZ_score --model_name orca2 \
# --granularity 3G \
# --cuda_device 6 \
# --weight_rouge 0.3 \
# --weight_bert 0.7 \
# --weight_contradiction 1 \
# --weight_min 0.0 \
# --weight_mean 1.0"

# "python fizz_coarse/fizz_coarse.py \
# --input_path data/data_aggrefact/aggre_fact_xsum_sota.json \
# --output_path results_coarse/xsum_tune/coarse_wr0p3_wb0p7_wc1_wm0p3_ww0p7.json \
# --doc_label document --summary_label claim --label_label label --score_column FIZZ_score --model_name orca2 \
# --granularity 3G \
# --cuda_device 6 \
# --weight_rouge 0.3 \
# --weight_bert 0.7 \
# --weight_contradiction 1 \
# --weight_min 0.3 \
# --weight_mean 0.7"

# "python fizz_LLM/fizz_original.py \
# --input_path data/data_aggrefact/aggre_fact_xsum_sota.json \
# --output_path results_LLM/xsum_tune/LLM_AGGRE_original.json \
# --doc_label document --summary_label claim --label_label label --score_column FIZZ_score --model_name orca2 \
# --granularity 3G \
# --cuda_device 6 \
# --mode original \
# --weight_rouge 0.0 \
# --weight_bert 1.0 \
# --base 2.0 \
# --exponent_delta 0.2"




)

# 각 스크립트 실행 및 시간 측정
for script in "${scripts[@]}"; do
    START_TIME=$(date +%s)

    # 스크립트 실행
    echo "실행 중: $script"
    eval $script

    END_TIME=$(date +%s)
    ELAPSED_TIME=$((END_TIME - START_TIME))

    # 로그 파일에 기록
    echo "경로: $script" >> $LOG_FILE
    echo "실행 시간: ${ELAPSED_TIME}초" >> $LOG_FILE
    echo "------------------------------------" >> $LOG_FILE
done

echo "모든 스크립트 실행 완료!" >> $LOG_FILE
echo "전체 종료 시간: $(date)" >> $LOG_FILE
echo "===== 실행 시간 기록 종료 =====" >> $LOG_FILE

echo "로그 파일이 생성되었습니다: $LOG_FILE"
