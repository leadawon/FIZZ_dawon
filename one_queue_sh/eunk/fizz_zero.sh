#!/bin/bash




python fizz_infuse/fizz_original.py \
--input_path data/data_aggrefact/aggre_fact_xsum_sota.json \
--output_path results/eunk/original_xsum_infuse.json \
--granularity 3G \
--cuda_device 0 \
--mode original \
--weight_rouge 0.0 \
--weight_bert 1.0 \
--base 2.0 \
--exponent_delta 0.2 \
--doc_label document \
--summary_label claim \
--label_label label \
--score_column FIZZ_score \
--model_name orca2






