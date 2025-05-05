import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from nltk.tokenize import sent_tokenize
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')
from rouge_score import rouge_scorer
from bert_score import BERTScorer
import os
import math
import torch.nn.functional as F
model_map = {
    "tals": {"model_ckpt": "tals/albert-xlarge-vitaminc-mnli"}
}

class AtomicFactScorer:
    def __init__(self, model_name="tals", granularity="3G", device="cuda"):
        assert granularity in ["1G", "2G", "3G", "4G"], "Wrong granularity %s" % (granularity)
        assert model_name in model_map.keys(), "Wrong model name: `%s`" % (model_name)

        self.granularity = granularity
        self.gran = int(granularity[0]) + 1
        self.model_name = model_name
        self.model_ckpt = model_map[self.model_name]["model_ckpt"]
        self.model = None
        self.device = device

        ## dawon code

        self.weight_rouge = 0.3#weight_rouge
        self.weight_bert = 0.3#weight_bert
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        self.bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)

    def load_lm(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_ckpt,
                                                       use_fast=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_ckpt)
        self.model.to(self.device)

    # from https://github.com/tingofurro/summac/
    def split_sentences(self, text):
        sentences = sent_tokenize(text)
        sentences = [sent for sent in sentences if len(sent)>10]
        return sentences
    

    # fizz original code
    # def atomic_facts_scoring(self, original_text, decomposed_text):
    #     self.load_lm()
    #     original_sentences_list = self.split_sentences(original_text)
    #     decomposed_sentences_list = self.split_sentences(decomposed_text)
        
    #     max_scores = []
    #     for decomposed_sentence in decomposed_sentences_list:
    #         decomposed_sentence_scores = [[], [], []]

    #         for original_sentence in original_sentences_list:
    #             features = self.tokenizer([original_sentence.strip()],
    #                                     [decomposed_sentence.strip()],
    #                                     padding=True,
    #                                     truncation=True,
    #                                     return_tensors="pt").to(self.device)
    #             self.model.eval()
    #             with torch.no_grad():
    #                 logits = self.model(**features).logits
    #                 scores = torch.nn.functional.softmax(logits, dim=-1)
    #                 evid_score = np.array(scores[0][0].cpu()).item()
    #                 conts_score = np.array(scores[0][1].cpu()).item()
    #                 neuts_score = np.array(scores[0][2].cpu()).item()

    #                 decomposed_sentence_scores[0].append(evid_score)
    #                 decomposed_sentence_scores[1].append(conts_score)
    #                 decomposed_sentence_scores[2].append(neuts_score)
            
    #         max_evid_score = max(decomposed_sentence_scores[0])
    #         max_evid_idx = decomposed_sentence_scores[0].index(max_evid_score)

    #         if decomposed_sentence_scores[0][max_evid_idx] > decomposed_sentence_scores[1][max_evid_idx] and decomposed_sentence_scores[0][max_evid_idx] > decomposed_sentence_scores[2][max_evid_idx]:
    #             max_scores.append(max_evid_score)
    #         else:
    #             temp_scores = []
    #             expanded_original_sentences = []

    #             expanded_gran_idx_list = get_combinations(list(range(len(original_sentences_list))), self.gran, max_evid_idx)
    #             for gran_idx_list in expanded_gran_idx_list:
    #                 new_original_sentences = ""
    #                 for gran_idx in gran_idx_list:
    #                     new_original_sentences += original_sentences_list[gran_idx] + " "
    #                 expanded_original_sentences.append(new_original_sentences)
                
    #             for expanded_original_sentence in expanded_original_sentences:
    #                 features = self.tokenizer([expanded_original_sentence.strip()],
    #                                           [decomposed_sentence.strip()],
    #                                           padding=True,
    #                                           truncation=True,
    #                                           return_tensors="pt").to(self.device)
    #                 self.model.eval()
    #                 with torch.no_grad():
    #                     logits = self.model(**features).logits
    #                     scores = torch.nn.functional.softmax(logits, dim=-1)
    #                     evid_score = np.array(scores[0][0].cpu()).item()
    #                     temp_scores.append(evid_score)
                
    #             max_temp_score = max(temp_scores)
    #             max_scores.append(max(max_evid_score, max_temp_score))
        
    #     min_max_score = min(max_scores)
    #     # min_idx = max_scores.index(min_max_score)

    #     return min_max_score
    







    # e로 ranking + fenice scoring (e-c) , np.min, np.mean 적용 짜증나게 np.mean한거 성능 오름.... 하............

    
    # def atomic_facts_scoring(self, original_text, decomposed_text):
    #     self.load_lm()
    #     original_sentences_list = self.split_sentences(original_text)
    #     decomposed_sentences_list = self.split_sentences(decomposed_text)

    #     max_scores = []

    #     for decomposed_sentence in decomposed_sentences_list:
    #         ecn_ranking_scores = []
    #         entailment_scores = []
    #         contradiction_scores = []
    #         neutral_scores = []

    #         # 1. 모든 original 문장에 대해 NLI 계산
    #         for original_sentence in original_sentences_list:
    #             features = self.tokenizer(
    #                 [original_sentence.strip()],
    #                 [decomposed_sentence.strip()],
    #                 padding=True,
    #                 truncation=True,
    #                 return_tensors="pt"
    #             ).to(self.device)

    #             self.model.eval()
    #             with torch.no_grad():
    #                 logits = self.model(**features).logits
    #                 probs = torch.nn.functional.softmax(logits, dim=-1)
    #                 entailment = probs[0][0].cpu().item()
    #                 contradiction = probs[0][1].cpu().item()
    #                 neutral = probs[0][2].cpu().item()
    #                 #ecn_score = entailment + contradiction - neutral
    #                 ecn_score = entailment #+ contradiction

    #                 ecn_ranking_scores.append(ecn_score)
    #                 entailment_scores.append(entailment)
    #                 contradiction_scores.append(contradiction)
    #                 neutral_scores.append(neutral)

    #         # 2. ECN 기준으로 최고 문장 선택 (단, entailment 점수는 따로 보관)
    #         max_ecn_idx = int(np.argmax(ecn_ranking_scores))
    #         max_entailment = entailment_scores[max_ecn_idx]
    #         max_contradiction = contradiction_scores[max_ecn_idx]
    #         max_neutral = neutral_scores[max_ecn_idx]

    #         # 3. dominant 판단: entailment가 가장 클 경우
    #         if max_entailment > max_contradiction and max_entailment > max_neutral:
    #             max_scores.append(max_entailment - max_contradiction)
    #         else:
    #             # 문맥 확장 조합 생성
    #             expanded_gran_idx_list = get_combinations(
    #                 list(range(len(original_sentences_list))),
    #                 self.gran,
    #                 max_ecn_idx
    #             )

    #             expanded_scores = []
    #             for gran_idx_list in expanded_gran_idx_list:
    #                 new_original = " ".join(original_sentences_list[i] for i in gran_idx_list)
    #                 features = self.tokenizer(
    #                     [new_original.strip()],
    #                     [decomposed_sentence.strip()],
    #                     padding=True,
    #                     truncation=True,
    #                     return_tensors="pt"
    #                 ).to(self.device)

    #                 with torch.no_grad():
    #                     logits = self.model(**features).logits
    #                     probs = torch.nn.functional.softmax(logits, dim=-1)
    #                     entailment = probs[0][0].cpu().item()
    #                     contradiction = probs[0][1].cpu().item()
    #                     neutral = probs[0][2].cpu().item()
    #                     #ecn_score = entailment + contradiction - neutral
    #                     ecn_score = entailment #+ contradiction
    #                     expanded_scores.append((ecn_score, entailment - contradiction)) # 여기 조심해라!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    #             # 확장 문맥들 중 최고 ECN
    #             best_ecn_from_expansion, best_entailment_contradiction_from_expansion = max(expanded_scores, key=lambda x: x[0])

    #             # 최종 비교도 ECN 기준
    #             if ecn_ranking_scores[max_ecn_idx] >= best_ecn_from_expansion:
    #                 max_scores.append(max_entailment - max_contradiction)
    #             else:
    #                 max_scores.append(best_entailment_contradiction_from_expansion)

    #     return np.mean(max_scores)








    

    # e로 ranking + infuse expansion + fenice scoring (e-c) , np.mean 
    
    # def atomic_facts_scoring(self, original_text, decomposed_text):
    #     self.load_lm()
    #     original_sentences_list = self.split_sentences(original_text)
    #     decomposed_sentences_list = self.split_sentences(decomposed_text)

    #     max_scores = []

    #     for decomposed_sentence in decomposed_sentences_list:
    #         ecn_ranking_scores = []
    #         entailment_scores = []
    #         contradiction_scores = []
    #         neutral_scores = []

    #         for original_sentence in original_sentences_list:
    #             features = self.tokenizer(
    #                 [original_sentence.strip()],
    #                 [decomposed_sentence.strip()],
    #                 padding=True,
    #                 truncation=True,
    #                 return_tensors="pt"
    #             ).to(self.device)

    #             self.model.eval()
    #             with torch.no_grad():
    #                 logits = self.model(**features).logits
    #                 probs = torch.nn.functional.softmax(logits, dim=-1)
    #                 entailment = probs[0][0].cpu().item()
    #                 contradiction = probs[0][1].cpu().item()
    #                 neutral = probs[0][2].cpu().item()
    #                 ecn_score = entailment  # or entailment + contradiction

    #                 ecn_ranking_scores.append(ecn_score)
    #                 entailment_scores.append(entailment)
    #                 contradiction_scores.append(contradiction)
    #                 neutral_scores.append(neutral)

    #         # dominant 판단
    #         max_ecn_idx = int(np.argmax(ecn_ranking_scores))
    #         max_entailment = entailment_scores[max_ecn_idx]
    #         max_contradiction = contradiction_scores[max_ecn_idx]
    #         max_neutral = neutral_scores[max_ecn_idx]

    #         if max_entailment > max_contradiction and max_entailment > max_neutral:
    #             max_scores.append(max_entailment - max_contradiction)
    #         else:
    #             # ecn_ranking 기준 정렬된 인덱스
    #             ranked_indices = sorted(range(len(ecn_ranking_scores)), key=lambda i: ecn_ranking_scores[i], reverse=True)
    #             concat_sentences = []
    #             prev_neutral = None
    #             selected_entailment = None
    #             selected_contradiction = None

    #             for i, idx in enumerate(ranked_indices):
    #                 concat_sentences.append(original_sentences_list[idx])
    #                 expanded_context = " ".join(concat_sentences)

    #                 features = self.tokenizer(
    #                     [expanded_context.strip()],
    #                     [decomposed_sentence.strip()],
    #                     padding=True,
    #                     truncation=True,
    #                     return_tensors="pt"
    #                 ).to(self.device)

    #                 with torch.no_grad():
    #                     logits = self.model(**features).logits
    #                     probs = torch.nn.functional.softmax(logits, dim=-1)
    #                     entailment = probs[0][0].cpu().item()
    #                     contradiction = probs[0][1].cpu().item()
    #                     neutral = probs[0][2].cpu().item()

    #                 if i == 0:
    #                     prev_neutral = neutral
    #                     selected_entailment = entailment
    #                     selected_contradiction = contradiction
    #                 else:
    #                     if neutral > prev_neutral:
    #                         break
    #                     else:
    #                         prev_neutral = neutral
    #                         selected_entailment = entailment
    #                         selected_contradiction = contradiction

    #             max_scores.append(selected_entailment - selected_contradiction)

    #     return np.mean(max_scores)



    # # ranking + infuse expansion(종료조건 엔트로피) + fenice scoring (e-c) , np.mean 
    # def atomic_facts_scoring_with_entropy(self, original_text, decomposed_text):
    #     self.load_lm()
    #     original_sentences_list = self.split_sentences(original_text)
    #     decomposed_sentences_list = self.split_sentences(decomposed_text)

    #     max_scores = []

    #     for decomposed_sentence in decomposed_sentences_list:
    #         ecn_ranking_scores = []
    #         entailment_probs = []

    #         for original_sentence in original_sentences_list:
    #             features = self.tokenizer(
    #                 [original_sentence.strip()],
    #                 [decomposed_sentence.strip()],
    #                 padding=True,
    #                 truncation=True,
    #                 return_tensors="pt"
    #             ).to(self.device)

    #             self.model.eval()
    #             with torch.no_grad():
    #                 logits = self.model(**features).logits
    #                 probs = torch.nn.functional.softmax(logits, dim=-1).squeeze().cpu().tolist()
    #                 entailment_probs.append(probs)
    #                 ecn_ranking_scores.append(probs[0])  # entailment

    #         ranked_indices = sorted(range(len(ecn_ranking_scores)), key=lambda i: ecn_ranking_scores[i], reverse=True)
    #         concat_sentences = []
    #         selected_entailment = None
    #         selected_contradiction = None
    #         prev_entropy = None

    #         for i, idx in enumerate(ranked_indices):
    #             concat_sentences.append(original_sentences_list[idx])
    #             expanded_context = " ".join(concat_sentences)

    #             features = self.tokenizer(
    #                 [expanded_context.strip()],
    #                 [decomposed_sentence.strip()],
    #                 padding=True,
    #                 truncation=True,
    #                 return_tensors="pt"
    #             ).to(self.device)

    #             with torch.no_grad():
    #                 logits = self.model(**features).logits
    #                 probs = torch.nn.functional.softmax(logits, dim=-1).squeeze().cpu().tolist()
    #                 entropy = -sum([x * math.log(x + 1e-10) for x in probs])
    #                 entailment = probs[0]
    #                 contradiction = probs[1]

    #             if i == 0:
    #                 prev_entropy = entropy
    #                 selected_entailment = entailment
    #                 selected_contradiction = contradiction
    #             else:
    #                 if entropy > prev_entropy:
    #                     break
    #                 else:
    #                     prev_entropy = entropy
    #                     selected_entailment = entailment
    #                     selected_contradiction = contradiction

    #         max_scores.append(selected_entailment - selected_contradiction)

    #     return np.mean(max_scores)


    # e ranking + infuse expansion(종료조건 bert+rouge) + fenice scoring (e-c) , np.mean 
    # def atomic_facts_scoring_with_similarity(self, original_text, decomposed_text):
    #     self.load_lm()
    #     original_sentences_list = self.split_sentences(original_text)
    #     decomposed_sentences_list = self.split_sentences(decomposed_text)

    #     max_scores = []

    #     for decomposed_sentence in decomposed_sentences_list:
    #         ecn_ranking_scores = []
    #         entailment_probs = []

    #         for original_sentence in original_sentences_list:
    #             features = self.tokenizer(
    #                 [original_sentence.strip()],
    #                 [decomposed_sentence.strip()],
    #                 padding=True,
    #                 truncation=True,
    #                 return_tensors="pt"
    #             ).to(self.device)

    #             self.model.eval()
    #             with torch.no_grad():
    #                 logits = self.model(**features).logits
    #                 probs = torch.nn.functional.softmax(logits, dim=-1).squeeze().cpu().tolist()
    #                 entailment_probs.append(probs)
    #                 ecn_ranking_scores.append(probs[0])  # entailment

    #         ranked_indices = sorted(range(len(ecn_ranking_scores)), key=lambda i: ecn_ranking_scores[i], reverse=True)
    #         concat_sentences = []
    #         selected_entailment = None
    #         selected_contradiction = None
    #         prev_similarity = None

    #         for i, idx in enumerate(ranked_indices):
    #             concat_sentences.append(original_sentences_list[idx])
    #             expanded_context = " ".join(concat_sentences)

    #             features = self.tokenizer(
    #                 [expanded_context.strip()],
    #                 [decomposed_sentence.strip()],
    #                 padding=True,
    #                 truncation=True,
    #                 return_tensors="pt"
    #             ).to(self.device)

    #             with torch.no_grad():
    #                 logits = self.model(**features).logits
    #                 probs = torch.nn.functional.softmax(logits, dim=-1).squeeze().cpu().tolist()
    #                 entailment = probs[0]
    #                 contradiction = probs[1]

    #             rouge_score = self.rouge_scorer.score(decomposed_sentence, expanded_context)['rougeL'].fmeasure
    #             _, _, F1 = self.bert_scorer.score([expanded_context], [decomposed_sentence])
    #             bert_score = F1.item()
    #             similarity = self.weight_rouge * rouge_score + self.weight_bert * bert_score

    #             if i == 0:
    #                 prev_similarity = similarity
    #                 selected_entailment = entailment
    #                 selected_contradiction = contradiction
    #             else:
    #                 if similarity < prev_similarity:
    #                     break
    #                 else:
    #                     prev_similarity = similarity
    #                     selected_entailment = entailment
    #                     selected_contradiction = contradiction

    #         max_scores.append(selected_entailment - selected_contradiction)

    #     return np.mean(max_scores)


# ranking + fizz/infuse expansion + fenice scoring (e-c) , np.mean 이거 모든 측면에서 best는 아니지만 이걸 내 sota로 설정정

    # def atomic_facts_scoring(self, original_text, decomposed_text):
    #     self.load_lm()
    #     original_sentences_list = self.split_sentences(original_text)
    #     decomposed_sentences_list = self.split_sentences(decomposed_text)

    #     max_scores = []

    #     for decomposed_sentence in decomposed_sentences_list:
    #         ecn_ranking_scores = []
    #         entailment_scores = []
    #         contradiction_scores = []
    #         neutral_scores = []

    #         # Step 1: NLI 계산
    #         for original_sentence in original_sentences_list:
    #             features = self.tokenizer(
    #                 [original_sentence.strip()],
    #                 [decomposed_sentence.strip()],
    #                 padding=True,
    #                 truncation=True,
    #                 return_tensors="pt"
    #             ).to(self.device)

    #             self.model.eval()
    #             with torch.no_grad():
    #                 logits = self.model(**features).logits
    #                 probs = torch.nn.functional.softmax(logits, dim=-1)
    #                 entailment = probs[0][0].item()
    #                 contradiction = probs[0][1].item()
    #                 neutral = probs[0][2].item()
    #                 ecn_score = entailment

    #                 ecn_ranking_scores.append(ecn_score)
    #                 entailment_scores.append(entailment)
    #                 contradiction_scores.append(contradiction)
    #                 neutral_scores.append(neutral)

    #         # Step 2: dominant 판단
    #         max_ecn_idx = int(np.argmax(ecn_ranking_scores))
    #         max_entailment = entailment_scores[max_ecn_idx]
    #         max_contradiction = contradiction_scores[max_ecn_idx]
    #         max_neutral = neutral_scores[max_ecn_idx]

    #         if max_entailment > max_contradiction and max_entailment > max_neutral:
    #             max_scores.append(max_entailment - max_contradiction)
    #         else:
    #             ### 방법 1: gran 기반 context 확장 ###
    #             expanded_gran_idx_list = get_combinations(
    #                 list(range(len(original_sentences_list))),
    #                 self.gran,
    #                 max_ecn_idx
    #             )

    #             gran_expanded_scores = []
    #             for gran_idx_list in expanded_gran_idx_list:
    #                 new_context = " ".join(original_sentences_list[i] for i in gran_idx_list)
    #                 features = self.tokenizer(
    #                     [new_context.strip()],
    #                     [decomposed_sentence.strip()],
    #                     padding=True,
    #                     truncation=True,
    #                     return_tensors="pt"
    #                 ).to(self.device)

    #                 with torch.no_grad():
    #                     logits = self.model(**features).logits
    #                     probs = torch.nn.functional.softmax(logits, dim=-1)
    #                     entailment = probs[0][0].item()
    #                     contradiction = probs[0][1].item()
                    
    #                 gran_expanded_scores.append(entailment - contradiction)

    #             best_gran_score = max(gran_expanded_scores) if gran_expanded_scores else -1

    #             ### 방법 2: ecn 내림차순 정렬 후 progressive 확장 (neutral 증가 시 종료) ###
    #             ranked_indices = sorted(range(len(ecn_ranking_scores)), key=lambda i: ecn_ranking_scores[i], reverse=True)
    #             concat_sentences = []
    #             prev_neutral = None
    #             selected_entailment = None
    #             selected_contradiction = None

    #             for i, idx in enumerate(ranked_indices):
    #                 concat_sentences.append(original_sentences_list[idx])
    #                 expanded_context = " ".join(concat_sentences)

    #                 features = self.tokenizer(
    #                     [expanded_context.strip()],
    #                     [decomposed_sentence.strip()],
    #                     padding=True,
    #                     truncation=True,
    #                     return_tensors="pt"
    #                 ).to(self.device)

    #                 with torch.no_grad():
    #                     logits = self.model(**features).logits
    #                     probs = torch.nn.functional.softmax(logits, dim=-1)
    #                     entailment = probs[0][0].item()
    #                     contradiction = probs[0][1].item()
    #                     neutral = probs[0][2].item()

    #                 if i == 0:
    #                     prev_neutral = neutral
    #                     selected_entailment = entailment
    #                     selected_contradiction = contradiction
    #                 else:
    #                     if neutral > prev_neutral:
    #                         break
    #                     prev_neutral = neutral
    #                     selected_entailment = entailment
    #                     selected_contradiction = contradiction

    #             ranked_expansion_score = selected_entailment - selected_contradiction

    #             ### 최종 선택: 두 확장 방식 중 더 좋은 score ###
    #             final_score = max(best_gran_score, ranked_expansion_score)
    #             max_scores.append(final_score)

    #     return np.mean(max_scores)






# # ranking + fizz/infuse expansion + fenice scoring (e-c) , np.mean 이거 모든 측면에서 best는 아니지만 이걸 내 sota로 설정정

    # def atomic_facts_scoring(self, original_text, decomposed_text):
    #     self.load_lm()
    #     original_sentences_list = self.split_sentences(original_text)
    #     decomposed_sentences_list = self.split_sentences(decomposed_text)

    #     max_scores = []

    #     for decomposed_sentence in decomposed_sentences_list:
    #         ecn_ranking_scores = []
    #         entailment_scores = []
    #         contradiction_scores = []
    #         neutral_scores = []

    #         # Step 1: NLI 계산
    #         for original_sentence in original_sentences_list:
    #             features = self.tokenizer(
    #                 [original_sentence.strip()],
    #                 [decomposed_sentence.strip()],
    #                 padding=True,
    #                 truncation=True,
    #                 return_tensors="pt"
    #             ).to(self.device)

    #             self.model.eval()
    #             with torch.no_grad():
    #                 logits = self.model(**features).logits
    #                 probs = torch.nn.functional.softmax(logits, dim=-1)
    #                 entailment = probs[0][0].item()
    #                 contradiction = probs[0][1].item()
    #                 neutral = probs[0][2].item()
    #                 ecn_score = entailment

    #                 ecn_ranking_scores.append(ecn_score)
    #                 entailment_scores.append(entailment)
    #                 contradiction_scores.append(contradiction)
    #                 neutral_scores.append(neutral)

    #         # Step 2: dominant 판단
    #         max_ecn_idx = int(np.argmax(ecn_ranking_scores))
    #         max_entailment = entailment_scores[max_ecn_idx]
    #         max_contradiction = contradiction_scores[max_ecn_idx]
    #         max_neutral = neutral_scores[max_ecn_idx]

    #         if max_entailment > max_contradiction and max_entailment > max_neutral:
    #             max_scores.append(max_entailment - max_contradiction)
    #         else:
    #             ### 방법 1: gran 기반 context 확장 ###
    #             expanded_gran_idx_list = get_combinations(
    #                 list(range(len(original_sentences_list))),
    #                 self.gran,
    #                 max_ecn_idx
    #             )

    #             gran_expanded_scores = []
    #             for gran_idx_list in expanded_gran_idx_list:
    #                 new_context = " ".join(original_sentences_list[i] for i in gran_idx_list)
    #                 features = self.tokenizer(
    #                     [new_context.strip()],
    #                     [decomposed_sentence.strip()],
    #                     padding=True,
    #                     truncation=True,
    #                     return_tensors="pt"
    #                 ).to(self.device)

    #                 with torch.no_grad():
    #                     logits = self.model(**features).logits
    #                     probs = torch.nn.functional.softmax(logits, dim=-1)
    #                     entailment = probs[0][0].item()
    #                     contradiction = probs[0][1].item()
                    
    #                 gran_expanded_scores.append(entailment - contradiction)

    #             best_gran_score = max(gran_expanded_scores) if gran_expanded_scores else -1

    #             ### 방법 2: ecn 내림차순 정렬 후 progressive 확장 (neutral 증가 시 종료) ###
    #             ranked_indices = sorted(range(len(ecn_ranking_scores)), key=lambda i: ecn_ranking_scores[i], reverse=True)
    #             concat_sentences = []
    #             prev_neutral = None
    #             selected_entailment = None
    #             selected_contradiction = None

    #             for i, idx in enumerate(ranked_indices):
    #                 concat_sentences.append(original_sentences_list[idx])
    #                 expanded_context = " ".join(concat_sentences)

    #                 features = self.tokenizer(
    #                     [expanded_context.strip()],
    #                     [decomposed_sentence.strip()],
    #                     padding=True,
    #                     truncation=True,
    #                     return_tensors="pt"
    #                 ).to(self.device)

    #                 with torch.no_grad():
    #                     logits = self.model(**features).logits
    #                     probs = torch.nn.functional.softmax(logits, dim=-1)
    #                     entailment = probs[0][0].item()
    #                     contradiction = probs[0][1].item()
    #                     neutral = probs[0][2].item()

    #                 if i == 0:
    #                     prev_neutral = neutral
    #                     selected_entailment = entailment
    #                     selected_contradiction = contradiction
    #                 else:
    #                     if neutral > prev_neutral:
    #                         break
    #                     prev_neutral = neutral
    #                     selected_entailment = entailment
    #                     selected_contradiction = contradiction

    #             ranked_expansion_score = selected_entailment - selected_contradiction

    #             ### 최종 선택: 두 확장 방식 중 더 좋은 score ###
    #             final_score = max(best_gran_score, ranked_expansion_score)
    #             max_scores.append(final_score)

    #     return np.mean(max_scores)

    # # ranking + fizz/infuse expansion + fizz scoring (e) , np.min 이거 모든 측면에서 best는 아니지만 이걸 내 sota로 설정정

    # def atomic_facts_scoring(self, original_text, decomposed_text):
    #     self.load_lm()
    #     print("ffizzninfuse min ablated")
    #     original_sentences_list = self.split_sentences(original_text)
    #     decomposed_sentences_list = self.split_sentences(decomposed_text)

    #     max_scores = []

    #     for decomposed_sentence in decomposed_sentences_list:
    #         ecn_ranking_scores = []
    #         entailment_scores = []
    #         contradiction_scores = []
    #         neutral_scores = []

    #         # Step 1: NLI 계산
    #         for original_sentence in original_sentences_list:
    #             features = self.tokenizer(
    #                 [original_sentence.strip()],
    #                 [decomposed_sentence.strip()],
    #                 padding=True,
    #                 truncation=True,
    #                 return_tensors="pt"
    #             ).to(self.device)

    #             self.model.eval()
    #             with torch.no_grad():
    #                 logits = self.model(**features).logits
    #                 probs = torch.nn.functional.softmax(logits, dim=-1)
    #                 entailment = probs[0][0].item()
    #                 contradiction = probs[0][1].item()
    #                 neutral = probs[0][2].item()
    #                 ecn_score = entailment

    #                 ecn_ranking_scores.append(ecn_score)
    #                 entailment_scores.append(entailment)
    #                 contradiction_scores.append(contradiction)
    #                 neutral_scores.append(neutral)

    #         # Step 2: dominant 판단
    #         max_ecn_idx = int(np.argmax(ecn_ranking_scores))
    #         max_entailment = entailment_scores[max_ecn_idx]
    #         max_contradiction = contradiction_scores[max_ecn_idx]
    #         max_neutral = neutral_scores[max_ecn_idx]

    #         if max_entailment > max_contradiction and max_entailment > max_neutral:
    #             max_scores.append(max_entailment )#- max_contradiction)
    #         else:
    #             ### 방법 1: gran 기반 context 확장 ###
    #             expanded_gran_idx_list = get_combinations(
    #                 list(range(len(original_sentences_list))),
    #                 self.gran,
    #                 max_ecn_idx
    #             )

    #             gran_expanded_scores = []
    #             for gran_idx_list in expanded_gran_idx_list:
    #                 new_context = " ".join(original_sentences_list[i] for i in gran_idx_list)
    #                 features = self.tokenizer(
    #                     [new_context.strip()],
    #                     [decomposed_sentence.strip()],
    #                     padding=True,
    #                     truncation=True,
    #                     return_tensors="pt"
    #                 ).to(self.device)

    #                 with torch.no_grad():
    #                     logits = self.model(**features).logits
    #                     probs = torch.nn.functional.softmax(logits, dim=-1)
    #                     entailment = probs[0][0].item()
    #                     contradiction = probs[0][1].item()
                    
    #                 gran_expanded_scores.append(entailment )#- contradiction)

    #             best_gran_score = max(gran_expanded_scores) if gran_expanded_scores else -1

    #             ### 방법 2: ecn 내림차순 정렬 후 progressive 확장 (neutral 증가 시 종료) ###
    #             ranked_indices = sorted(range(len(ecn_ranking_scores)), key=lambda i: ecn_ranking_scores[i], reverse=True)
    #             concat_sentences = []
    #             prev_neutral = None
    #             selected_entailment = None
    #             selected_contradiction = None

    #             for i, idx in enumerate(ranked_indices):
    #                 concat_sentences.append(original_sentences_list[idx])
    #                 expanded_context = " ".join(concat_sentences)

    #                 features = self.tokenizer(
    #                     [expanded_context.strip()],
    #                     [decomposed_sentence.strip()],
    #                     padding=True,
    #                     truncation=True,
    #                     return_tensors="pt"
    #                 ).to(self.device)

    #                 with torch.no_grad():
    #                     logits = self.model(**features).logits
    #                     probs = torch.nn.functional.softmax(logits, dim=-1)
    #                     entailment = probs[0][0].item()
    #                     contradiction = probs[0][1].item()
    #                     neutral = probs[0][2].item()

    #                 if i == 0:
    #                     prev_neutral = neutral
    #                     selected_entailment = entailment
    #                     selected_contradiction = contradiction
    #                 else:
    #                     if neutral > prev_neutral:
    #                         break
    #                     prev_neutral = neutral
    #                     selected_entailment = entailment
    #                     selected_contradiction = contradiction

    #             ranked_expansion_score = selected_entailment #- selected_contradiction

    #             ### 최종 선택: 두 확장 방식 중 더 좋은 score ###
    #             final_score = max(best_gran_score, ranked_expansion_score)
    #             max_scores.append(final_score)

    #     return min(max_scores)

# e ranking dawon fifuse expansion ,e-c scoring, 낫베드이고 노벨티도 있음. 중요한거는 ablation study인데...
    # def atomic_facts_scoring(self, original_text, decomposed_text):
    #     self.load_lm()
    #     original_sentences_list = self.split_sentences(original_text)
    #     decomposed_sentences_list = self.split_sentences(decomposed_text)

    #     max_scores = []

    #     for decomposed_sentence in decomposed_sentences_list:
    #         ecn_ranking_scores = []
    #         entailment_scores = []
    #         contradiction_scores = []
    #         neutral_scores = []

    #         for original_sentence in original_sentences_list:
    #             features = self.tokenizer(
    #                 [original_sentence.strip()],
    #                 [decomposed_sentence.strip()],
    #                 padding=True,
    #                 truncation=True,
    #                 return_tensors="pt"
    #             ).to(self.device)

    #             self.model.eval()
    #             with torch.no_grad():
    #                 logits = self.model(**features).logits
    #                 probs = torch.nn.functional.softmax(logits, dim=-1)
    #                 entailment = probs[0][0].item()
    #                 contradiction = probs[0][1].item()
    #                 neutral = probs[0][2].item()
    #                 ecn_score = entailment

    #             ecn_ranking_scores.append(ecn_score)
    #             entailment_scores.append(entailment)
    #             contradiction_scores.append(contradiction)
    #             neutral_scores.append(neutral)

    #         max_ecn_idx = int(np.argmax(ecn_ranking_scores))
    #         max_entailment = entailment_scores[max_ecn_idx]
    #         max_contradiction = contradiction_scores[max_ecn_idx]
    #         max_neutral = neutral_scores[max_ecn_idx]

    #         if max_entailment > max_contradiction and max_entailment > max_neutral:
    #             max_scores.append(max_entailment - max_contradiction)
    #         else:
    #             ranked_indices = sorted(range(len(entailment_scores)), key=lambda i: entailment_scores[i], reverse=True)

    #             best_combo_score = -float('inf')
    #             best_combo = []
    #             prev_neutral = None

    #             for i, idx in enumerate(ranked_indices):
    #                 gran_combos = get_combinations(list(range(len(original_sentences_list))), self.gran, idx)

    #                 best_gran_score = -float('inf')
    #                 best_gran_combo = []
    #                 best_gran_neutral = None

    #                 for combo in gran_combos:
    #                     context = " ".join(original_sentences_list[j] for j in combo)
    #                     features = self.tokenizer(
    #                         [context.strip()],
    #                         [decomposed_sentence.strip()],
    #                         padding=True,
    #                         truncation=True,
    #                         return_tensors="pt"
    #                     ).to(self.device)

    #                     with torch.no_grad():
    #                         logits = self.model(**features).logits
    #                         probs = torch.nn.functional.softmax(logits, dim=-1)
    #                         entailment = probs[0][0].item()
    #                         contradiction = probs[0][1].item()
    #                         neutral = probs[0][2].item()

    #                     score = entailment - contradiction
    #                     if score > best_gran_score:
    #                         best_gran_score = score
    #                         best_gran_combo = combo
    #                         best_gran_neutral = neutral

    #                 if i == 0:
    #                     prev_neutral = best_gran_neutral
    #                     best_combo = best_gran_combo
    #                     best_combo_score = best_gran_score
    #                 else:
    #                     if best_gran_neutral > prev_neutral:
    #                         break
    #                     combined = sorted(set(best_combo + best_gran_combo))
    #                     context = " ".join(original_sentences_list[j] for j in combined)
    #                     features = self.tokenizer(
    #                         [context.strip()],
    #                         [decomposed_sentence.strip()],
    #                         padding=True,
    #                         truncation=True,
    #                         return_tensors="pt"
    #                     ).to(self.device)

    #                     with torch.no_grad():
    #                         logits = self.model(**features).logits
    #                         probs = torch.nn.functional.softmax(logits, dim=-1)
    #                         entailment = probs[0][0].item()
    #                         contradiction = probs[0][1].item()

    #                     score = entailment - contradiction
    #                     if score > best_combo_score:
    #                         best_combo = combined
    #                         best_combo_score = score
    #                     prev_neutral = best_gran_neutral

    #             max_scores.append(best_combo_score)

    #     return np.mean(max_scores)



# e ranking dawon fifuse expansion ,e-c scoring, 노벨티 좀더 추가한 버전. expansion phase원래 optional이었는데 madatory하게 바꿈
    # def atomic_facts_scoring(self, original_text, decomposed_text):
    #     self.load_lm()
    #     original_sentences_list = self.split_sentences(original_text)
    #     decomposed_sentences_list = self.split_sentences(decomposed_text)

    #     max_scores = []

    #     for decomposed_sentence in decomposed_sentences_list:
    #         ecn_ranking_scores = []
    #         entailment_scores = []
    #         contradiction_scores = []
    #         neutral_scores = []

    #         for original_sentence in original_sentences_list:
    #             features = self.tokenizer(
    #                 [original_sentence.strip()],
    #                 [decomposed_sentence.strip()],
    #                 padding=True,
    #                 truncation=True,
    #                 return_tensors="pt"
    #             ).to(self.device)

    #             self.model.eval()
    #             with torch.no_grad():
    #                 logits = self.model(**features).logits
    #                 probs = torch.nn.functional.softmax(logits, dim=-1)
    #                 entailment = probs[0][0].item()
    #                 contradiction = probs[0][1].item()
    #                 neutral = probs[0][2].item()
    #                 ecn_score = entailment

    #             ecn_ranking_scores.append(ecn_score)
    #             entailment_scores.append(entailment)
    #             contradiction_scores.append(contradiction)
    #             neutral_scores.append(neutral)

    #         max_ecn_idx = int(np.argmax(ecn_ranking_scores))
    #         max_entailment = entailment_scores[max_ecn_idx]
    #         max_contradiction = contradiction_scores[max_ecn_idx]
    #         max_neutral = neutral_scores[max_ecn_idx]

    #         if max_entailment > max_contradiction and max_entailment > max_neutral and False:
    #             max_scores.append(max_entailment - max_contradiction)
    #         else:
    #             ranked_indices = sorted(range(len(entailment_scores)), key=lambda i: entailment_scores[i], reverse=True)

    #             best_combo_score = -float('inf')
    #             best_combo = []
    #             prev_neutral = None

    #             for i, idx in enumerate(ranked_indices):
    #                 gran_combos = get_combinations(list(range(len(original_sentences_list))), self.gran, idx)

    #                 best_gran_score = -float('inf')
    #                 best_gran_combo = []
    #                 best_gran_neutral = None

    #                 for combo in gran_combos:
    #                     context = " ".join(original_sentences_list[j] for j in combo)
    #                     features = self.tokenizer(
    #                         [context.strip()],
    #                         [decomposed_sentence.strip()],
    #                         padding=True,
    #                         truncation=True,
    #                         return_tensors="pt"
    #                     ).to(self.device)

    #                     with torch.no_grad():
    #                         logits = self.model(**features).logits
    #                         probs = torch.nn.functional.softmax(logits, dim=-1)
    #                         entailment = probs[0][0].item()
    #                         contradiction = probs[0][1].item()
    #                         neutral = probs[0][2].item()

    #                     score = entailment - contradiction
    #                     if score > best_gran_score:
    #                         best_gran_score = score
    #                         best_gran_combo = combo
    #                         best_gran_neutral = neutral

    #                 if i == 0:
    #                     prev_neutral = best_gran_neutral
    #                     best_combo = best_gran_combo
    #                     best_combo_score = best_gran_score
    #                 else:
    #                     if best_gran_neutral > prev_neutral:
    #                         break
    #                     combined = sorted(set(best_combo + best_gran_combo))
    #                     context = " ".join(original_sentences_list[j] for j in combined)
    #                     features = self.tokenizer(
    #                         [context.strip()],
    #                         [decomposed_sentence.strip()],
    #                         padding=True,
    #                         truncation=True,
    #                         return_tensors="pt"
    #                     ).to(self.device)

    #                     with torch.no_grad():
    #                         logits = self.model(**features).logits
    #                         probs = torch.nn.functional.softmax(logits, dim=-1)
    #                         entailment = probs[0][0].item()
    #                         contradiction = probs[0][1].item()

    #                     score = entailment - contradiction
    #                     if score > best_combo_score:
    #                         best_combo = combined
    #                         best_combo_score = score
    #                     prev_neutral = best_gran_neutral

    #             max_scores.append(best_combo_score)

    #     return np.mean(max_scores)




# e-c ranking dawon fifuse expansion ,e - c scoring, np.mean 이거 ranking e-c로 하면 점수 많이 떨어짐
    # def atomic_facts_scoring(self, original_text, decomposed_text):
    #     self.load_lm()
    #     original_sentences_list = self.split_sentences(original_text)
    #     decomposed_sentences_list = self.split_sentences(decomposed_text)

    #     max_scores = []

    #     for decomposed_sentence in decomposed_sentences_list:
    #         ecn_ranking_scores = []
    #         entailment_scores = []
    #         contradiction_scores = []
    #         neutral_scores = []

    #         for original_sentence in original_sentences_list:
    #             features = self.tokenizer(
    #                 [original_sentence.strip()],
    #                 [decomposed_sentence.strip()],
    #                 padding=True,
    #                 truncation=True,
    #                 return_tensors="pt"
    #             ).to(self.device)

    #             self.model.eval()
    #             with torch.no_grad():
    #                 logits = self.model(**features).logits
    #                 probs = torch.nn.functional.softmax(logits, dim=-1)
    #                 entailment = probs[0][0].item()
    #                 contradiction = probs[0][1].item()
    #                 neutral = probs[0][2].item()
    #                 ecn_score = entailment - contradiction

    #             ecn_ranking_scores.append(ecn_score)
    #             entailment_scores.append(entailment)
    #             contradiction_scores.append(contradiction)
    #             neutral_scores.append(neutral)

    #         max_ecn_idx = int(np.argmax(ecn_ranking_scores))
    #         max_entailment = entailment_scores[max_ecn_idx]
    #         max_contradiction = contradiction_scores[max_ecn_idx]
    #         max_neutral = neutral_scores[max_ecn_idx]

    #         if max_entailment > max_contradiction and max_entailment > max_neutral:
    #             max_scores.append(max_entailment - max_contradiction)
    #         else:
    #             ranked_indices = sorted(range(len(ecn_ranking_scores)), key=lambda i: ecn_ranking_scores[i], reverse=True)

    #             best_combo_score = -float('inf')
    #             best_combo = []
    #             prev_neutral = None

    #             for i, idx in enumerate(ranked_indices):
    #                 gran_combos = get_combinations(list(range(len(original_sentences_list))), self.gran, idx)

    #                 best_gran_score = -float('inf')
    #                 best_gran_combo = []
    #                 best_gran_neutral = None

    #                 for combo in gran_combos:
    #                     context = " ".join(original_sentences_list[j] for j in combo)
    #                     features = self.tokenizer(
    #                         [context.strip()],
    #                         [decomposed_sentence.strip()],
    #                         padding=True,
    #                         truncation=True,
    #                         return_tensors="pt"
    #                     ).to(self.device)

    #                     with torch.no_grad():
    #                         logits = self.model(**features).logits
    #                         probs = torch.nn.functional.softmax(logits, dim=-1)
    #                         entailment = probs[0][0].item()
    #                         contradiction = probs[0][1].item()
    #                         neutral = probs[0][2].item()

    #                     score = entailment - contradiction
    #                     if score > best_gran_score:
    #                         best_gran_score = score
    #                         best_gran_combo = combo
    #                         best_gran_neutral = neutral

    #                 if i == 0:
    #                     prev_neutral = best_gran_neutral
    #                     best_combo = best_gran_combo
    #                     best_combo_score = best_gran_score
    #                 else:
    #                     if best_gran_neutral > prev_neutral:
    #                         break
    #                     combined = sorted(set(best_combo + best_gran_combo))
    #                     context = " ".join(original_sentences_list[j] for j in combined)
    #                     features = self.tokenizer(
    #                         [context.strip()],
    #                         [decomposed_sentence.strip()],
    #                         padding=True,
    #                         truncation=True,
    #                         return_tensors="pt"
    #                     ).to(self.device)

    #                     with torch.no_grad():
    #                         logits = self.model(**features).logits
    #                         probs = torch.nn.functional.softmax(logits, dim=-1)
    #                         entailment = probs[0][0].item()
    #                         contradiction = probs[0][1].item()

    #                     score = entailment - contradiction
    #                     if score > best_combo_score:
    #                         best_combo = combined
    #                         best_combo_score = score
    #                     prev_neutral = best_gran_neutral

    #             max_scores.append(best_combo_score)

    #     return np.mean(max_scores)
    


# e ranking dawon fifuse expansion(rouge, bertscore) ,e-c scoring, np.mean
    # def atomic_facts_scoring(self, original_text, decomposed_text):
    #     self.load_lm()
    #     original_sentences_list = self.split_sentences(original_text)
    #     decomposed_sentences_list = self.split_sentences(decomposed_text)

    #     max_scores = []

    #     for decomposed_sentence in decomposed_sentences_list:
    #         ecn_ranking_scores = []
    #         entailment_scores = []
    #         contradiction_scores = []
    #         neutral_scores = []

    #         rouge_scores = []
    #         bert_scores = []

    #         for original_sentence in original_sentences_list:
    #             features = self.tokenizer(
    #                 [original_sentence.strip()],
    #                 [decomposed_sentence.strip()],
    #                 padding=True,
    #                 truncation=True,
    #                 return_tensors="pt"
    #             ).to(self.device)

    #             self.model.eval()
    #             with torch.no_grad():
    #                 logits = self.model(**features).logits
    #                 probs = torch.nn.functional.softmax(logits, dim=-1)
    #                 entailment = probs[0][0].item()
    #                 contradiction = probs[0][1].item()
    #                 neutral = probs[0][2].item()

    #             # rouge_score = self.rouge_scorer.score(decomposed_sentence, original_sentence)['rougeL'].fmeasure
    #             _, _, F1 = self.bert_scorer.score([original_sentence], [decomposed_sentence])
    #             bert_score = F1.item()

    #             ecn_score = entailment + self.weight_bert * bert_score

    #             ecn_ranking_scores.append(ecn_score)
    #             entailment_scores.append(entailment)
    #             contradiction_scores.append(contradiction)
    #             neutral_scores.append(neutral)

    #         max_ecn_idx = int(np.argmax(ecn_ranking_scores))
    #         max_entailment = entailment_scores[max_ecn_idx]
    #         max_contradiction = contradiction_scores[max_ecn_idx]
    #         max_neutral = neutral_scores[max_ecn_idx]

    #         if max_entailment > max_contradiction and max_entailment > max_neutral:
    #             max_scores.append(max_entailment - max_contradiction)
    #         else:
    #             ranked_indices = sorted(range(len(ecn_ranking_scores)), key=lambda i: entailment_scores[i], reverse=True)
    #             best_combo_score = -float('inf')
    #             best_combo = []
    #             prev_neutral = None

    #             for i, idx in enumerate(ranked_indices):
    #                 gran_combos = get_combinations(list(range(len(original_sentences_list))), self.gran, idx)

    #                 best_gran_score = -float('inf')
    #                 best_gran_combo = []
    #                 best_gran_neutral = None

    #                 for combo in gran_combos:
    #                     context = " ".join(original_sentences_list[j] for j in combo)
    #                     features = self.tokenizer(
    #                         [context.strip()],
    #                         [decomposed_sentence.strip()],
    #                         padding=True,
    #                         truncation=True,
    #                         return_tensors="pt"
    #                     ).to(self.device)

    #                     with torch.no_grad():
    #                         logits = self.model(**features).logits
    #                         probs = torch.nn.functional.softmax(logits, dim=-1)
    #                         entailment = probs[0][0].item()
    #                         contradiction = probs[0][1].item()
    #                         neutral = probs[0][2].item()
    #                     # rouge_score = self.rouge_scorer.score(decomposed_sentence, context)['rougeL'].fmeasure
    #                     _, _, F1 = self.bert_scorer.score([context], [decomposed_sentence])
    #                     bert_score = F1.item()
    #                     score = entailment + self.weight_bert * bert_score

    #                     if score > best_gran_score:
    #                         best_gran_score = score
    #                         best_gran_combo = combo
    #                         best_gran_neutral = neutral

    #                 if i == 0:
    #                     prev_neutral = best_gran_neutral
    #                     best_combo = best_gran_combo
    #                     best_combo_score = best_gran_score
    #                 else:
    #                     if best_gran_neutral > prev_neutral:
    #                         break
    #                     combined = sorted(set(best_combo + best_gran_combo))
    #                     context = " ".join(original_sentences_list[j] for j in combined)
    #                     features = self.tokenizer(
    #                         [context.strip()],
    #                         [decomposed_sentence.strip()],
    #                         padding=True,
    #                         truncation=True,
    #                         return_tensors="pt"
    #                     ).to(self.device)

    #                     with torch.no_grad():
    #                         logits = self.model(**features).logits
    #                         probs = torch.nn.functional.softmax(logits, dim=-1)
    #                         entailment = probs[0][0].item()
    #                         contradiction = probs[0][1].item()

    #                     rouge_score = self.rouge_scorer.score(decomposed_sentence, context)['rougeL'].fmeasure
    #                     _, _, F1 = self.bert_scorer.score([context], [decomposed_sentence])
    #                     bert_score = F1.item()
    #                     score = entailment + self.weight_rouge * rouge_score + self.weight_bert * bert_score

    #                     if score > best_combo_score:
    #                         best_combo = combined
    #                         best_combo_score = score
    #                     prev_neutral = best_gran_neutral

    #             max_scores.append(best_combo_score)

    #     return np.mean(max_scores)





# e로 ranking + fenice scoring (e-c) , minus softmin-0.2

    
    # def atomic_facts_scoring(self, original_text, decomposed_text, base, exponent_delta):
    #     self.load_lm()
    #     original_sentences_list = self.split_sentences(original_text)
    #     decomposed_sentences_list = self.split_sentences(decomposed_text)

    #     max_scores = []

    #     for decomposed_sentence in decomposed_sentences_list:
    #         ecn_ranking_scores = []
    #         entailment_scores = []
    #         contradiction_scores = []
    #         neutral_scores = []

    #         # 1. 모든 original 문장에 대해 NLI 계산
    #         for original_sentence in original_sentences_list:
    #             features = self.tokenizer(
    #                 [original_sentence.strip()],
    #                 [decomposed_sentence.strip()],
    #                 padding=True,
    #                 truncation=True,
    #                 return_tensors="pt"
    #             ).to(self.device)

    #             self.model.eval()
    #             with torch.no_grad():
    #                 logits = self.model(**features).logits
    #                 probs = torch.nn.functional.softmax(logits, dim=-1)
    #                 entailment = probs[0][0].cpu().item()
    #                 contradiction = probs[0][1].cpu().item()
    #                 neutral = probs[0][2].cpu().item()
    #                 #ecn_score = entailment + contradiction - neutral
    #                 ecn_score = entailment #+ contradiction

    #                 ecn_ranking_scores.append(ecn_score)
    #                 entailment_scores.append(entailment)
    #                 contradiction_scores.append(contradiction)
    #                 neutral_scores.append(neutral)

    #         # 2. ECN 기준으로 최고 문장 선택 (단, entailment 점수는 따로 보관)
    #         max_ecn_idx = int(np.argmax(ecn_ranking_scores))
    #         max_entailment = entailment_scores[max_ecn_idx]
    #         max_contradiction = contradiction_scores[max_ecn_idx]
    #         max_neutral = neutral_scores[max_ecn_idx]

    #         # 3. dominant 판단: entailment가 가장 클 경우
    #         if max_entailment > max_contradiction and max_entailment > max_neutral:
    #             max_scores.append(max_entailment - max_contradiction)
    #         else:
    #             # 문맥 확장 조합 생성
    #             expanded_gran_idx_list = get_combinations(
    #                 list(range(len(original_sentences_list))),
    #                 self.gran,
    #                 max_ecn_idx
    #             )

    #             expanded_scores = []
    #             for gran_idx_list in expanded_gran_idx_list:
    #                 new_original = " ".join(original_sentences_list[i] for i in gran_idx_list)
    #                 features = self.tokenizer(
    #                     [new_original.strip()],
    #                     [decomposed_sentence.strip()],
    #                     padding=True,
    #                     truncation=True,
    #                     return_tensors="pt"
    #                 ).to(self.device)

    #                 with torch.no_grad():
    #                     logits = self.model(**features).logits
    #                     probs = torch.nn.functional.softmax(logits, dim=-1)
    #                     entailment = probs[0][0].cpu().item()
    #                     contradiction = probs[0][1].cpu().item()
    #                     neutral = probs[0][2].cpu().item()
    #                     #ecn_score = entailment + contradiction - neutral
    #                     ecn_score = entailment #+ contradiction
    #                     expanded_scores.append((ecn_score, entailment - contradiction)) 

    #             # 확장 문맥들 중 최고 ECN
    #             best_ecn_from_expansion, best_entailment_contradiction_from_expansion = max(expanded_scores, key=lambda x: x[0])

    #             # 최종 비교도 ECN 기준
    #             if ecn_ranking_scores[max_ecn_idx] >= best_ecn_from_expansion:
    #                 max_scores.append(max_entailment - max_contradiction)
    #             else:
    #                 max_scores.append(best_entailment_contradiction_from_expansion)

    #     transformed_scores = [-math.pow(base, -(s-exponent_delta)) for s in max_scores]
    #     final_score = sum(transformed_scores) / len(transformed_scores)

    #     # 로그 작성
    #     # log_path = "./log/scoring_last.log"
    #     # os.makedirs(os.path.dirname(log_path), exist_ok=True)


    #     # with open(log_path, "a", encoding="utf-8") as f:
    #     #     f.write("[Max Scores]\n")
    #     #     for score in max_scores:
    #     #         f.write(f"{score:.6f}\n")

    #     #     mean_raw_score = np.mean(max_scores)
    #     #     f.write(f"\n[Mean Raw Score]\n{mean_raw_score:.6f}\n")

    #     #     f.write("\n[Transformed Scores]\n")
    #     #     for tscore in transformed_scores:
    #     #         f.write(f"{tscore:.6f}\n")

    #     #     f.write(f"\n[Final Score (Transformed Mean)]\n{final_score:.6f}\n")
    #     #     f.write(f"\n[Original Decomposed Text]\n{decomposed_text}\n")
    #     #     f.write("="*40 + "\n\n")

    #     return final_score






# e로 ranking + fenice scoring (e-c) , interactive minus softmin 쌉구림

    
    # def atomic_facts_scoring(self, original_text, decomposed_text, base, exponent_delta):
    #     self.load_lm()
    #     original_sentences_list = self.split_sentences(original_text)
    #     decomposed_sentences_list = self.split_sentences(decomposed_text)

    #     max_scores = []

    #     for decomposed_sentence in decomposed_sentences_list:
    #         ecn_ranking_scores = []
    #         entailment_scores = []
    #         contradiction_scores = []
    #         neutral_scores = []

    #         # 1. 모든 original 문장에 대해 NLI 계산
    #         for original_sentence in original_sentences_list:
    #             features = self.tokenizer(
    #                 [original_sentence.strip()],
    #                 [decomposed_sentence.strip()],
    #                 padding=True,
    #                 truncation=True,
    #                 return_tensors="pt"
    #             ).to(self.device)

    #             self.model.eval()
    #             with torch.no_grad():
    #                 logits = self.model(**features).logits
    #                 probs = torch.nn.functional.softmax(logits, dim=-1)
    #                 entailment = probs[0][0].cpu().item()
    #                 contradiction = probs[0][1].cpu().item()
    #                 neutral = probs[0][2].cpu().item()
    #                 ecn_score = entailment

    #             ecn_ranking_scores.append(ecn_score)
    #             entailment_scores.append(entailment)
    #             contradiction_scores.append(contradiction)
    #             neutral_scores.append(neutral)

    #         # 2. ECN 기준으로 최고 문장 선택
    #         max_ecn_idx = int(np.argmax(ecn_ranking_scores))
    #         max_entailment = entailment_scores[max_ecn_idx]
    #         max_contradiction = contradiction_scores[max_ecn_idx]
    #         max_neutral = neutral_scores[max_ecn_idx]

    #         # 3. dominant 판단
    #         if max_entailment > max_contradiction and max_entailment > max_neutral:
    #             max_scores.append(max_entailment - max_contradiction)
    #         else:
    #             expanded_gran_idx_list = get_combinations(
    #                 list(range(len(original_sentences_list))),
    #                 self.gran,
    #                 max_ecn_idx
    #             )

    #             expanded_scores = []
    #             for gran_idx_list in expanded_gran_idx_list:
    #                 new_original = " ".join(original_sentences_list[i] for i in gran_idx_list)
    #                 features = self.tokenizer(
    #                     [new_original.strip()],
    #                     [decomposed_sentence.strip()],
    #                     padding=True,
    #                     truncation=True,
    #                     return_tensors="pt"
    #                 ).to(self.device)

    #                 with torch.no_grad():
    #                     logits = self.model(**features).logits
    #                     probs = torch.nn.functional.softmax(logits, dim=-1)
    #                     entailment = probs[0][0].cpu().item()
    #                     contradiction = probs[0][1].cpu().item()
    #                     neutral = probs[0][2].cpu().item()
    #                     ecn_score = entailment
    #                     expanded_scores.append((ecn_score, entailment - contradiction))

    #             best_ecn_from_expansion, best_entailment_contradiction_from_expansion = max(
    #                 expanded_scores, key=lambda x: x[0])

    #             if ecn_ranking_scores[max_ecn_idx] >= best_ecn_from_expansion:
    #                 max_scores.append(max_entailment - max_contradiction)
    #             else:
    #                 max_scores.append(best_entailment_contradiction_from_expansion)

    #     # 지수 함수 적용을 위한 base 조정
    #     num_sentences = len(decomposed_sentences_list)
    #     base_adjusted = base + (num_sentences - 3) * base_flex

    #     transformed_scores = [-math.pow(base_adjusted, -(s - exponent_delta)) for s in max_scores]
    #     final_score = sum(transformed_scores) / len(transformed_scores)

    #     return final_score



# e로 ranking + fenice scoring (e-c) , minus softmin-0.2

    
    # def atomic_facts_scoring(self, original_text, decomposed_text, base, exponent_delta):
    #     self.load_lm()
    #     original_sentences_list = self.split_sentences(original_text)
    #     decomposed_sentences_list = self.split_sentences(decomposed_text)

    #     max_scores = []

    #     for decomposed_sentence in decomposed_sentences_list:
    #         ecn_ranking_scores = []
    #         entailment_scores = []
    #         contradiction_scores = []
    #         neutral_scores = []

    #         # 1. 모든 original 문장에 대해 NLI 계산
    #         for original_sentence in original_sentences_list:
    #             features = self.tokenizer(
    #                 [original_sentence.strip()],
    #                 [decomposed_sentence.strip()],
    #                 padding=True,
    #                 truncation=True,
    #                 return_tensors="pt"
    #             ).to(self.device)

    #             self.model.eval()
    #             with torch.no_grad():
    #                 logits = self.model(**features).logits
    #                 probs = torch.nn.functional.softmax(logits, dim=-1)
    #                 entailment = probs[0][0].cpu().item()
    #                 contradiction = probs[0][1].cpu().item()
    #                 neutral = probs[0][2].cpu().item()
    #                 #ecn_score = entailment + contradiction - neutral
    #                 ecn_score = entailment #+ contradiction

    #                 ecn_ranking_scores.append(ecn_score)
    #                 entailment_scores.append(entailment)
    #                 contradiction_scores.append(contradiction)
    #                 neutral_scores.append(neutral)

    #         # 2. ECN 기준으로 최고 문장 선택 (단, entailment 점수는 따로 보관)
    #         max_ecn_idx = int(np.argmax(ecn_ranking_scores))
    #         max_entailment = entailment_scores[max_ecn_idx]
    #         max_contradiction = contradiction_scores[max_ecn_idx]
    #         max_neutral = neutral_scores[max_ecn_idx]

    #         # 3. dominant 판단: entailment가 가장 클 경우
    #         if max_entailment > max_contradiction and max_entailment > max_neutral:
    #             max_scores.append(max_entailment - max_contradiction)
    #         else:
    #             # 문맥 확장 조합 생성
    #             expanded_gran_idx_list = get_combinations(
    #                 list(range(len(original_sentences_list))),
    #                 self.gran,
    #                 max_ecn_idx
    #             )

    #             expanded_scores = []
    #             for gran_idx_list in expanded_gran_idx_list:
    #                 new_original = " ".join(original_sentences_list[i] for i in gran_idx_list)
    #                 features = self.tokenizer(
    #                     [new_original.strip()],
    #                     [decomposed_sentence.strip()],
    #                     padding=True,
    #                     truncation=True,
    #                     return_tensors="pt"
    #                 ).to(self.device)

    #                 with torch.no_grad():
    #                     logits = self.model(**features).logits
    #                     probs = torch.nn.functional.softmax(logits, dim=-1)
    #                     entailment = probs[0][0].cpu().item()
    #                     contradiction = probs[0][1].cpu().item()
    #                     neutral = probs[0][2].cpu().item()
    #                     #ecn_score = entailment + contradiction - neutral
    #                     ecn_score = entailment #+ contradiction
    #                     expanded_scores.append((ecn_score, entailment - contradiction)) 

    #             # 확장 문맥들 중 최고 ECN
    #             best_ecn_from_expansion, best_entailment_contradiction_from_expansion = max(expanded_scores, key=lambda x: x[0])

    #             # 최종 비교도 ECN 기준
    #             if ecn_ranking_scores[max_ecn_idx] >= best_ecn_from_expansion:
    #                 max_scores.append(max_entailment - max_contradiction)
    #             else:
    #                 max_scores.append(best_entailment_contradiction_from_expansion)

    #     #transformed_scores = [-math.pow(base, -(s-exponent_delta)) for s in max_scores]
    #     #final_score = sum(transformed_scores) / len(transformed_scores)
    #     final_score = 0.5 * np.mean(max_scores) + 0.5 * np.min(max_scores)


    #     # 로그 작성
    #     # log_path = "./log/scoring_last.log"
    #     # os.makedirs(os.path.dirname(log_path), exist_ok=True)


    #     # with open(log_path, "a", encoding="utf-8") as f:
    #     #     f.write("[Max Scores]\n")
    #     #     for score in max_scores:
    #     #         f.write(f"{score:.6f}\n")

    #     #     mean_raw_score = np.mean(max_scores)
    #     #     f.write(f"\n[Mean Raw Score]\n{mean_raw_score:.6f}\n")

    #     #     f.write("\n[Transformed Scores]\n")
    #     #     for tscore in transformed_scores:
    #     #         f.write(f"{tscore:.6f}\n")

    #     #     f.write(f"\n[Final Score (Transformed Mean)]\n{final_score:.6f}\n")
    #     #     f.write(f"\n[Original Decomposed Text]\n{decomposed_text}\n")
    #     #     f.write("="*40 + "\n\n")

    #     return final_score









def is_consecutive_by_one(numbers):
    for i in range(1, len(numbers)):
        if abs(numbers[i] - numbers[i-1]) != 1:
            return False
    return True

def get_combinations(num_list, size, target):
    combination_list = []
    for i in range(1, size):
        combination = combinations(num_list, i)
        comb_list = list(combination)
        combination_list.extend(comb_list)
    
    possible_idx_list = []
    for combination in combination_list:
        idx_list = list(combination)
        if target in idx_list and is_consecutive_by_one(idx_list):
            possible_idx_list.append(idx_list)

    return possible_idx_list

def main():
    scorer = AtomicFactScorer(granularity="4G")
    # original = "lisa courtney, of hertfordshire, has spent most of her life collecting pokemon memorabilia."
    # atomic_facts = "Lisa Courtney is from Hertfordshire. Lisa Courtney has spent most of her life collecting Pokémon memorabilia."
    
    original = "todd phillips, a pit crew member was hit by francesco dracone, a car on sunday during the inaugural indycar grand prix of louisiana . todd phillips , a front - outside tire changer for dayle coyne racing , was injuried when todd phillips was struck by francesco dracone, the car of francesco dracone , who had come in on lap 25 for tires and fuel . francesco dracone, dracone spun while exiting francesco dracone's put box , clipping phillips ' leg . todd phillips, tire changer todd phillips , a front - outside tire changer for dayle coyne racing , was injuried when todd phillips was struck by francesco dracone, the car of francesco dracone , who had come in on lap 25 for tires and fuel phillips was taken to the infield care center for treatment where todd phillips has received stitches for a cut on todd phillips's leg and has been released . many cars struggled with traction during the race after rain had fallen for hours leading up to the green flag . francesco dracone, dracone did not finish the race and wound up 23rd . francesco dracone ( 19 ) , of italy , spins out on turn one during the indycar grand prix of louisiana auto race in avondale on sunday"
    atomic_facts = "Todd Phillips is a tire changer. Todd Phillips was injured. Todd Phillips was struck by a car. The car that struck Todd Phillips was driven by Francesco Dracone. The event where Todd Phillips was struck by the car was the inaugural IndyCar Grand Prix of Louisiana. The date of the event was Sunday. Dracone came in on lap 25. Dracone needed tires and fuel. Dracone spun while exiting the pit box. Todd Phillips was taken to the infield care center. Todd Phillips received stitches for a cut on Todd Phillips's leg. Todd Phillips has been released. "

    score = scorer.atomic_facts_scoring(original, atomic_facts)
    print(score)

if __name__ == "__main__":
    main()