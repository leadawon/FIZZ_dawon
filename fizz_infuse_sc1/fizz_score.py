import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from atomic_fact_decomposition import AtomicFactDecomposer
from atomic_fact_filtering import AtomicFactFilterer
from atomic_fact_scoring import AtomicFactScorer
from datasets import Dataset
from tqdm import tqdm
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='fairy')

parser.add_argument('--input_path', type=str, default='data/aggre_fact_sota.csv')
parser.add_argument('--output_path', type=str, default='data/output.csv')
parser.add_argument('--doc_label', type=str, default='doc')
parser.add_argument('--summary_label', type=str, default='summary')
parser.add_argument('--atomic_facts_column', type=str, default='atomic_facts')
parser.add_argument('--score_column', type=str, default='FIZZ_score')
parser.add_argument('--model_name', type=str, default='orca2')
parser.add_argument('--granularity', type=str, default='3G')

args = parser.parse_args()

def main():
    dataset_input_path = args.input_path
    df = pd.read_csv(r'{}'.format(dataset_input_path), index_col=0)
    dataset = Dataset.from_pandas(df, preserve_index=False)

    decomposer = AtomicFactDecomposer(model_name=args.model_name)
    filterer = AtomicFactFilterer()
    scorer = AtomicFactScorer(granularity=args.granularity)

    docs = dataset[args.doc_label]
    summaries = dataset[args.summary_label]

    filtered_atomic_facts_list = []
    score_list = []
    n = dataset.num_rows

    for i in tqdm(range(n), desc="all", mininterval=0.01):
        doc = docs[i]
        summary = summaries[i]

        atomic_facts = decomposer.atomic_facts_decompose(summary)
        filtered_atomic_facts = filterer.atomic_facts_filtering(summary, atomic_facts)
        score = scorer.atomic_facts_scoring(doc, filtered_atomic_facts)

        filtered_atomic_facts_list.append(filtered_atomic_facts)
        score_list.append(score)

    # 중복 컬럼 제거
    if args.atomic_facts_column in dataset.column_names:
        dataset = dataset.remove_columns(args.atomic_facts_column)
    if args.score_column in dataset.column_names:
        dataset = dataset.remove_columns(args.score_column)

    # 컬럼 추가
    dataset = dataset.add_column(args.atomic_facts_column, filtered_atomic_facts_list)
    dataset = dataset.add_column(args.score_column, score_list)
    
    # DataFrame 변환 및 저장
    df_output = pd.DataFrame(dataset)
    df_output.to_csv(r'{}'.format(args.output_path))

if __name__ == "__main__":
    main()
