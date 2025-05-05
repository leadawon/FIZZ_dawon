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
    

    # # fizz original code
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
    

# fizz original + LLM aggregate
    def atomic_facts_scoring(self, original_text, decomposed_text, dawonclass):
        self.load_lm()
        original_sentences_list = self.split_sentences(original_text)
        decomposed_sentences_list = self.split_sentences(decomposed_text)

        

        sum_texts = []
        doc_texts = []
        scores = []

        for decomposed_sentence in decomposed_sentences_list:
            decomposed_sentence_scores = [[], [], []]

            for original_sentence in original_sentences_list:
                features = self.tokenizer(
                    [original_sentence.strip()],
                    [decomposed_sentence.strip()],
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)

                self.model.eval()
                with torch.no_grad():
                    logits = self.model(**features).logits
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                    evid_score = probs[0][0].cpu().item()
                    conts_score = probs[0][1].cpu().item()
                    neuts_score = probs[0][2].cpu().item()

                    decomposed_sentence_scores[0].append(evid_score)
                    decomposed_sentence_scores[1].append(conts_score)
                    decomposed_sentence_scores[2].append(neuts_score)

            max_evid_score = max(decomposed_sentence_scores[0])
            max_evid_idx = decomposed_sentence_scores[0].index(max_evid_score)

            if (decomposed_sentence_scores[0][max_evid_idx] > decomposed_sentence_scores[1][max_evid_idx] and
                decomposed_sentence_scores[0][max_evid_idx] > decomposed_sentence_scores[2][max_evid_idx]):
                # dominant 판단
                best_doc = original_sentences_list[max_evid_idx]
                best_score_tuple = (
                    decomposed_sentence_scores[0][max_evid_idx],
                    decomposed_sentence_scores[1][max_evid_idx],
                    decomposed_sentence_scores[2][max_evid_idx]
                )
            else:
                # 확장 문맥 조합 생성
                temp_scores = []
                temp_score_tuples = []
                expanded_gran_idx_list = get_combinations(
                    list(range(len(original_sentences_list))), self.gran, max_evid_idx
                )

                for gran_idx_list in expanded_gran_idx_list:
                    new_doc = " ".join(original_sentences_list[i] for i in gran_idx_list)
                    features = self.tokenizer(
                        [new_doc.strip()],
                        [decomposed_sentence.strip()],
                        padding=True,
                        truncation=True,
                        return_tensors="pt"
                    ).to(self.device)

                    with torch.no_grad():
                        logits = self.model(**features).logits
                        probs = torch.nn.functional.softmax(logits, dim=-1)
                        entailment = probs[0][0].cpu().item()
                        contradiction = probs[0][1].cpu().item()
                        neutral = probs[0][2].cpu().item()

                        temp_scores.append(entailment)
                        temp_score_tuples.append((entailment, contradiction, neutral))

                if temp_scores:
                    max_temp_score = max(temp_scores)
                    best_score_tuple = temp_score_tuples[temp_scores.index(max_temp_score)]
                    best_doc = " ".join(original_sentences_list[i] for i in expanded_gran_idx_list[temp_scores.index(max_temp_score)])
                else:
                    # fallback
                    best_score_tuple = (
                        decomposed_sentence_scores[0][max_evid_idx],
                        decomposed_sentence_scores[1][max_evid_idx],
                        decomposed_sentence_scores[2][max_evid_idx]
                    )
                    best_doc = original_sentences_list[max_evid_idx]

            # LLM에 넘길 정보 누적
            sum_texts.append(decomposed_sentence.strip())
            doc_texts.append([best_doc.strip()])
            scores.append([best_score_tuple])


        

        final_result = dawonclass.aggregate(sum_texts, doc_texts, scores)
        return final_result











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