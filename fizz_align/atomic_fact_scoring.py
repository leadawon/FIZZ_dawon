import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from nltk.tokenize import sent_tokenize
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')
from rouge_score import rouge_scorer
from bert_score import BERTScorer
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

        self.weight_rouge = 1#weight_rouge
        self.weight_bert = 1#weight_bert
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
    

    # #ecn score 코드
    # def atomic_facts_scoring(self, original_text, decomposed_text, modified_sentences):
    #     self.load_lm()
    #     original_sentences_list = self.split_sentences(original_text)
    #     decomposed_sentences_list = self.split_sentences(decomposed_text)

    #     max_scores = []
    #     for decomposed_sentence in decomposed_sentences_list:
    #         entailment_scores = []
    #         contradiction_scores = []
    #         neutral_scores = []

    #         for original_sentence, sent_type in modified_sentences:
    #             features = self.tokenizer([original_sentence.strip()],
    #                                     [decomposed_sentence.strip()],
    #                                     padding=True,
    #                                     truncation=True,
    #                                     return_tensors="pt").to(self.device)
    #             self.model.eval()
    #             with torch.no_grad():
    #                 logits = self.model(**features).logits
    #                 scores = torch.nn.functional.softmax(logits, dim=-1)
    #                 evid_score = scores[0][0].cpu().item()
    #                 conts_score = scores[0][1].cpu().item()
    #                 neuts_score = scores[0][2].cpu().item()

    #                 entailment_scores.append(evid_score)
    #                 contradiction_scores.append(conts_score)
    #                 neutral_scores.append(neuts_score)

    #         #ecn_scores = [e + c - n for e, c, n in zip(entailment_scores, contradiction_scores, neutral_scores)]
    #         ecn_scores = [e for e, c, n in zip(entailment_scores, contradiction_scores, neutral_scores)]
    #         max_ecn_idx = int(np.argmax(ecn_scores))
    #         max_entailment = entailment_scores[max_ecn_idx]
    #         max_contradiction = contradiction_scores[max_ecn_idx]
    #         max_neutral = neutral_scores[max_ecn_idx]

    #         if max_entailment > max_contradiction and max_entailment > max_neutral:
    #             max_scores.append(max_entailment)
    #         else:
    #             expanded_scores = []
    #             num_sentences = len(modified_sentences)
    #             for i in range(num_sentences):
    #                 expanded_contexts = []

    #                 if i > 0:
    #                     prev_sentence, prev_type = modified_sentences[i - 1]
    #                     cur_sentence, cur_type = modified_sentences[i]
    #                     if cur_type == "fact":
    #                         expanded_contexts.append(prev_sentence + " " + cur_sentence)

    #                 if i < num_sentences - 1:
    #                     cur_sentence, cur_type = modified_sentences[i]
    #                     next_sentence, next_type = modified_sentences[i + 1]
    #                     if cur_type == "fact" and next_type == "fact":
    #                         expanded_contexts.append(cur_sentence + " " + next_sentence)

    #                     if cur_type == "original" and next_type == "original":
    #                         expanded_contexts.append(cur_sentence + " " + next_sentence)

    #                 if i < num_sentences - 2:
    #                     cur_sentence, cur_type = modified_sentences[i]
    #                     next_sentence1, next_type1 = modified_sentences[i + 1]
    #                     next_sentence2, next_type2 = modified_sentences[i + 2]
    #                     if cur_type == "original" and next_type1 == "fact" and next_type2 == "fact":
    #                         expanded_contexts.append(cur_sentence + " " + next_sentence1 + " " + next_sentence2)
    #                     if cur_type == "fact" and next_type1 == "fact" and next_type2 == "original":
    #                         expanded_contexts.append(cur_sentence + " " + next_sentence1 + " " + next_sentence2)

    #                 for context in expanded_contexts:
    #                     features = self.tokenizer([context.strip()],
    #                                             [decomposed_sentence.strip()],
    #                                             padding=True,
    #                                             truncation=True,
    #                                             return_tensors="pt").to(self.device)
    #                     self.model.eval()
    #                     with torch.no_grad():
    #                         logits = self.model(**features).logits
    #                         scores = torch.nn.functional.softmax(logits, dim=-1)
    #                         evid_score = scores[0][0].cpu().item()
    #                         conts_score = scores[0][1].cpu().item()
    #                         neuts_score = scores[0][2].cpu().item()
    #                         #ecn_score = evid_score + conts_score - neuts_score
    #                         ecn_score = evid_score
    #                         expanded_scores.append((ecn_score, evid_score))

    #             if expanded_scores:
    #                 best_ecn_from_expansion, best_entailment_from_expansion = max(expanded_scores, key=lambda x: x[0])
    #                 if ecn_scores[max_ecn_idx] >= best_ecn_from_expansion:
    #                     max_scores.append(entailment_scores[max_ecn_idx])
    #                 else:
    #                     max_scores.append(best_entailment_from_expansion)
    #             else:
    #                 max_scores.append(entailment_scores[max_ecn_idx])

    #     return min(max_scores)


    def atomic_facts_scoring(self, original_text, decomposed_text, modified_sentences):
        self.load_lm()

        original_sentences_list = [sent for sent, typ, _ in modified_sentences if typ == "original"]
        decomposed_sentences_list = self.split_sentences(decomposed_text)

        max_scores = []

        for decomposed_sentence in decomposed_sentences_list:
            combined_scores = []
            entailment_scores = []
            contradiction_scores = []
            neutral_scores = []

            for original_sentence, sent_type in modified_sentences:
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
                    entailment = probs[0][0].cpu().item()
                    contradiction = probs[0][1].cpu().item()
                    neutral = probs[0][2].cpu().item()

                rouge_score = self.rouge_scorer.score(decomposed_sentence, original_sentence)['rougeL'].fmeasure
                P, R, F1 = self.bert_scorer.score([original_sentence], [decomposed_sentence])
                bert_score = F1.item()

                combined = entailment + self.weight_rouge * rouge_score + self.weight_bert * bert_score

                combined_scores.append(combined)
                entailment_scores.append(entailment)
                contradiction_scores.append(contradiction)
                neutral_scores.append(neutral)

            best_idx = int(np.argmax(combined_scores))
            best_entailment = entailment_scores[best_idx]
            best_contradiction = contradiction_scores[best_idx]
            best_neutral = neutral_scores[best_idx]

            if best_entailment > best_contradiction and best_entailment > best_neutral:
                max_scores.append(best_entailment)
            else:
                expanded_scores = []
                num_sentences = len(modified_sentences)
                for i in range(num_sentences):
                    expanded_contexts = []

                    if i > 0:
                        prev_sentence, prev_type = modified_sentences[i - 1]
                        cur_sentence, cur_type = modified_sentences[i]
                        if cur_type == "fact":
                            expanded_contexts.append(prev_sentence + " " + cur_sentence)

                    if i < num_sentences - 1:
                        cur_sentence, cur_type = modified_sentences[i]
                        next_sentence, next_type = modified_sentences[i + 1]
                        if cur_type == "fact" and next_type == "fact":
                            expanded_contexts.append(cur_sentence + " " + next_sentence)
                        if cur_type == "original" and next_type == "original":
                            expanded_contexts.append(cur_sentence + " " + next_sentence)

                    if i < num_sentences - 2:
                        cur_sentence, cur_type = modified_sentences[i]
                        next_sentence1, next_type1 = modified_sentences[i + 1]
                        next_sentence2, next_type2 = modified_sentences[i + 2]
                        if cur_type == "original" and next_type1 == "fact" and next_type2 == "fact":
                            expanded_contexts.append(cur_sentence + " " + next_sentence1 + " " + next_sentence2)
                        if cur_type == "fact" and next_type1 == "fact" and next_type2 == "original":
                            expanded_contexts.append(cur_sentence + " " + next_sentence1 + " " + next_sentence2)

                    for context in expanded_contexts:
                        features = self.tokenizer(
                            [context.strip()],
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

                        rouge_score = self.rouge_scorer.score(decomposed_sentence, context)['rougeL'].fmeasure
                        P, R, F1 = self.bert_scorer.score([context], [decomposed_sentence])
                        bert_score = F1.item()

                        combined = entailment + self.weight_rouge * rouge_score + self.weight_bert * bert_score
                        expanded_scores.append((combined, entailment))

                if expanded_scores:
                    best_combined_exp, best_entailment_exp = max(expanded_scores, key=lambda x: x[0])
                    if combined_scores[best_idx] >= best_combined_exp:
                        max_scores.append(best_entailment)
                    else:
                        max_scores.append(best_entailment_exp)
                else:
                    max_scores.append(best_entailment)

        return min(max_scores)





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