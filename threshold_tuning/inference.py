import torch
import numpy as np
import os
import logging
import sklearn.metrics 
from tqdm import tqdm
import pickle
logging.disable(logging.WARNING)
device = "cuda" if torch.cuda.is_available() else "cpu"
import pandas as pd

from transformers import BertTokenizer, BertForSequenceClassification
from transformers import RobertaForSequenceClassification, RobertaTokenizer


def load_nli_model(model_path):
     # bert일때 
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)   # tokenizer 뭐가 max length??
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3).to(device)
    ''' # roberta일때 
    model_name = "roberta-base"
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=3).to(device)
    '''
    state_dict = torch.load(model_path, map_location=device)
    new_state_dict = {k[7:]: v if k.startswith("module.") else v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()
    return tokenizer, model

def split_into_sentences(text):   #  이 부분 필요할 수도 
    return text.split(". ")

def get_nli_score(tokenizer, model, premise, hypothesis):
    inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1)
    entailment_score = probs[0][0].item()
    return entailment_score


def choose_best_threshold(labels, scores):
    best_bacc = 0.0
    best_thresh = 0.0
    thresholds = [np.percentile(scores, p) for p in np.arange(0, 100, 0.2)]
    for thresh in thresholds:
        preds = [1 if score > thresh else 0 for score in scores]
        bacc_score = sklearn.metrics.balanced_accuracy_score(labels, preds)
        if bacc_score >= best_bacc:
            best_bacc = bacc_score
            best_thresh = thresh
    return best_thresh, best_bacc

def from_score_to_pred(dataset, score_key):
    scores = [d[score_key] for d in dataset]
    labels = [d["label"] for d in dataset]
    thresh, _ = choose_best_threshold(labels, scores)

    pred_key = "pred_%s" % (score_key)
    for d in dataset:
        d[pred_key] = 1 if d[score_key] > thresh else 0

    return dataset, thresh

def evaluate_benchmark(model_paths, save_csv_path="evaluation_results.csv"):
    datasets = ["frank", "cogensumm", "xsumfaith", "polytope", 'factcc']
    results = []
    
    for model_path in tqdm(model_paths, desc="Evaluating Models"):
        tokenizer, model = load_nli_model(model_path) 
        
        for dataset_name in tqdm(datasets, desc=f"Processing {model_path}", leave=False):
            filename = f'benchmark/{dataset_name}_test.pkl'
            with open(filename,"rb") as fr:
                data = pickle.load(fr)

            val_filename = f'benchmark/{dataset_name}_val.pkl'
            with open(val_filename,"rb") as fr:
                val_data = pickle.load(fr)
            
            val_scores, val_labels = [], []
            for item in tqdm(val_data, desc=f"Validating {dataset_name}", leave=False):
                document_text = item['document']
                summary_text = item['claim']
                score = get_nli_score(tokenizer, model, document_text, summary_text)
                val_scores.append(score)
                val_labels.append(item["label"])

            best_threshold, _ = choose_best_threshold(val_labels, val_scores)

            test_scores, test_labels, test_preds = [], [], []
            for item in tqdm(data, desc=f"Testing {dataset_name}", leave=False):
                document_text = item['document']
                summary_text = item['claim']
                score = get_nli_score(tokenizer, model, document_text, summary_text)
                test_scores.append(score)
                test_labels.append(item["label"])
                test_preds.append(1 if score > best_threshold else 0)

            balanced_acc = sklearn.metrics.balanced_accuracy_score(test_labels, test_preds)
            precision = sklearn.metrics.precision_score(test_labels, test_preds, zero_division=0)
            recall = sklearn.metrics.recall_score(test_labels, test_preds, zero_division=0)
            f1 = sklearn.metrics.f1_score(test_labels, test_preds, zero_division=0)
            try:
                auc = sklearn.metrics.roc_auc_score(test_labels, test_scores)
            except ValueError:
                auc = float('nan')

            results.append({
                "model": os.path.basename(model_path),
                "dataset": dataset_name,
                "balanced_accuracy": balanced_acc,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "auc_roc": auc,
                "threshold": best_threshold
            })

            print(f"\n=== {os.path.basename(model_path)} | {dataset_name} ===")
            print(f"Balanced Accuracy: {balanced_acc:.4f}")
            print(f"Precision:         {precision:.4f}")
            print(f"Recall:            {recall:.4f}")
            print(f"F1 Score:          {f1:.4f}")
            print(f"AUC-ROC:           {auc:.4f}")

    # CSV 저장
    df = pd.DataFrame(results)
    df.to_csv(save_csv_path, index=False)
    print(f"\n📁 Evaluation results saved to {save_csv_path}")
    return results


# 모델 경로 리스트 
result_folder = "result_falsesum_margin0.1"
model_paths = [os.path.join(result_folder, f) for f in os.listdir(result_folder) if f.endswith(".pt")]

# 실행 및 저장
all_results = evaluate_benchmark(model_paths, save_csv_path="evaluation_results.csv")

