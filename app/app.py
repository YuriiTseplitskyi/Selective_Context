print('Loading dependencies...')
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
from typing import List, Tuple
import spacy
import numpy as np
import os
from dataclasses import dataclass
from nltk.tokenize import sent_tokenize, word_tokenize
import time

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

@dataclass
class LexicalUnits:
    unit_type: str
    text: List[str]
    self_info: List[float] = None

    def __add__(self, other):
        assert self.unit_type == other.unit_type, 'Cannot add two different unit types'
        return LexicalUnits(self.unit_type, self.text + other.text, self.self_info + other.self_info)
    
    def __radd__(self, other):
        if other == 0:
            return self
        return NotImplementedError()
    
    def add_to_head(self, token, self_info):
        return LexicalUnits(self.unit_type, [token] + self.text, [self_info] + self.self_info)
    
    def add_to_tail(self, token, self_info):
        return LexicalUnits(self.unit_type, self.text + [token], self.self_info + [self_info])

class SelectiveContext:

    def __init__(self, model_name = 'TheBloke/Llama-2-7B-Chat-GPTQ'):

        self.model_name = model_name
        self.device = DEVICE

        # this means we calculate self-information sentence by sentence
        self.sent_level_self_info = True

        self._prepare_phrase_tokenizer()
        self.sent_tokenize_pattern = r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s"
        self.phrase_mask_token = ''
        self.sent_mask_token = "<...some content omitted.>"
        self.keep_leading_word = False
        self.mask_token = ''
        self._prepare_model()
    
    def _prepare_phrase_tokenizer(self):
        # we use space to tokenize sentence into phrases
        # for English, we should use `spacy.load("en_core_web_sm").add_pipe('merge_noun_chunks')`
        # for Chinese, use `nlp = spacy.load('zh_core_web_sm')`` directly
        self.nlp = spacy.load("en_core_web_sm", disable=["ner"])
        self.nlp.add_pipe('merge_noun_chunks')

    def _prepare_model(self):
        """Load the language model and tokenizer from Hugging Face"""
        print(f"Loading model: {self.model_name}")
        
        # Use AutoTokenizer and AutoModelForCausalLM for generalized support
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Some models require legacy padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            print(f"Model {self.model_name} loaded successfully")
            
            # Get model's max context length
            if hasattr(self.model.config, "n_positions"):
                self.max_token_length = self.model.config.n_positions
            elif hasattr(self.model.config, "max_position_embeddings"):
                self.max_token_length = self.model.config.max_position_embeddings
            else:
                self.max_token_length = 1024  # Default fallback
                print(f"Warning: Could not determine model's maximum context length. Using default: {self.max_token_length}")
            
            self.get_self_information = self._get_self_info_via_hf_model
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def get_self_information(self, text: str) -> Tuple[List[str], List[float]]:
        # it takes text as input, and return a list of words and a list of self-information scores
        raise NotImplementedError

    def _get_self_info_via_hf_model(self, text: str) -> Tuple[List[str], List[float]]:
        """
        Generic function to calculate self-information using any Hugging Face model
        
        Includes caching to avoid recalculating for previously seen texts.
        """
        # Use cache if available
        if not hasattr(self, '_self_info_cache'):
            self._self_info_cache = {}
        
        # Check if result is in cache
        cache_key = text
        if cache_key in self._self_info_cache:
            return self._self_info_cache[cache_key]
        
        # Add model-specific prefixes/tags if needed
        if self.model_name.startswith('gpt2'):
            prefixed_text = f"<|endoftext|>{text}"
        elif "opt" in self.model_name:
            prefixed_text = f"</s>{text}"
        else:
            prefixed_text = text  # Default with no special prefix
        
        with torch.no_grad():
            # Tokenize the input text
            encoding = self.tokenizer(prefixed_text, add_special_tokens=False, return_tensors='pt')
            encoding = encoding.to(self.device)
            
            # Get model outputs
            outputs = self.model(**encoding)
            logits = outputs.logits
            
            # Calculate probabilities and self-information
            probs = torch.softmax(logits, dim=-1)
            self_info = -torch.log(probs)
            
            # Get input IDs for token decoding
            input_ids = encoding['input_ids']
            input_ids_expanded = input_ids[:, 1:].unsqueeze(-1)
            
            # Skip the prefix token in the result
            skip_tokens = 1  # Number of prefix tokens to skip
            tokens = [self.tokenizer.decode(token_id) for token_id in input_ids.squeeze().tolist()[skip_tokens:]]
            token_self_info = self_info[:, skip_tokens-1:-1].gather(-1, input_ids_expanded[:, skip_tokens:]).squeeze(-1).squeeze(0).tolist()
            
            # Store in cache
            result = (tokens, token_self_info)
            self._self_info_cache[cache_key] = result
            
            return result
    
    def _lexical_unit(self, sents):

        if self.sent_level_self_info:
            sent_self_info = []
            all_noun_phrases = []
            all_noun_phrases_info = []
            all_tokens = []
            all_token_self_info = []

            for sent in sents:
                # print(sent)
                tokens, self_info = self.get_self_information(sent)
                sent_self_info.append(np.mean(self_info))

                all_tokens.extend(tokens)
                all_token_self_info.extend(self_info)

                noun_phrases, noun_phrases_info = self._calculate_lexical_unit(tokens, self_info)

                # We need to add a space before the first noun phrase for every sentence except the first one
                if all_noun_phrases:
                    noun_phrases[0] = f" {noun_phrases[0]}"
                all_noun_phrases.extend(noun_phrases)
                all_noun_phrases_info.extend(noun_phrases_info)
            
            return [
                LexicalUnits('sent', text=sents, self_info=sent_self_info),
                LexicalUnits('phrase', text=all_noun_phrases, self_info=all_noun_phrases_info),
                LexicalUnits('token', text=all_tokens, self_info=all_token_self_info)
            ]
    
    def _calculate_lexical_unit(self, tokens, self_info):
        def _unit_info(tokens, self_info, units):
            current_unit_idx = 0
            current_position = 0
            unit_self_info = [[] for _ in range(len(units))]

            for idx, (token, info) in enumerate(zip(tokens, self_info)):
                current_position += len(token)
                if current_position == len(units[current_unit_idx]):
                    unit_self_info[current_unit_idx].append(info)
                    current_position = current_position - len(units[current_unit_idx])
                    current_unit_idx += 1
                elif current_position > len(units[current_unit_idx]):
                    counter_ = 1
                    current_position = current_position - len(units[current_unit_idx])
                    current_unit_idx += 1
                    while current_position >= len(units[current_unit_idx]):
                        counter_ += 1
                        current_position = current_position - len(units[current_unit_idx])
                        current_unit_idx += 1
                        if current_unit_idx >= len(units):
                            break
                    partial_info = info/counter_
                    for _ in range(counter_):
                        unit_self_info[(current_unit_idx-1) - _].append(partial_info)
                else:
                    if token == " ":
                        continue
                    unit_self_info[current_unit_idx].append(info)
            
            unit_self_info_ = [np.mean(info) for info in unit_self_info]
            return unit_self_info_
        
        def _noun_phrases(sent):
            noun_phrases = []
            doc = self.nlp(sent)
            for index, chunk in enumerate(doc):
                if index == 0:
                    noun_phrases.append(chunk.text)
                else:
                    noun_phrases.append(doc[index-1].whitespace_ + chunk.text)
            return noun_phrases

        if self.sent_level_self_info:
            # in this case, the self_info is for each sentence
            # we only need to calculate the self_info for each phrase

            sent = ''.join(tokens)
            # noun_phrases = [chunk.text for chunk in self.nlp(sent).noun_chunks]
            noun_phrases = _noun_phrases(sent)
            # noun_phrases[-1] = noun_phrases[-1] + ' '
            noun_phrases_info = _unit_info(tokens, self_info, noun_phrases)

            return noun_phrases, noun_phrases_info

    def beautify_context(self, context: str) -> str:
        context = re.sub(r"\s+", " ", context)
        return context

    def self_info_mask(self, sents: List[str], self_info: List[float], mask_level):
        # mask_level: mask sentences, phrases, or tokens
        sents_after_mask = []
        masked_sents = []
                
        self.ppl_threshold = np.nanpercentile(self_info, self.mask_ratio * 100)

        # if title is not None:
        #     with open(os.path.join(self.path, title+'_prob_token.tsv'), 'w', encoding='utf-8') as f:
        #         for token, info in zip(tokens, self_info):
        #             f.write(f"{token}\t{info}\n")
        #     with open(os.path.join(self.path, title+'_prob_sent.tsv'), 'w', encoding='utf-8') as f:
        #         for sent, info in zip(sents, sent_self_info):
        #             f.write(f"{sent}\n{info}\n\n")

        for sent, info in zip(sents, self_info):
            if info < self.ppl_threshold:
                masked_sents.append(sent)
                sents_after_mask.append(self.mask_a_sent(sent, mask_level))
            else:
                sents_after_mask.append(sent)
        masked_context = " ".join(sents_after_mask) if mask_level == 'sent' else "".join(sents_after_mask)
        
        return masked_context, masked_sents

    def mask_a_sent(self, sent, level):
        if level == 'phrase':
            return self.phrase_mask_token
        elif level == 'sent':
            if self.keep_leading_word:
                leading_few_words = " ".join(word_tokenize(sent)[:self.num_lead_words]) + " "
            else:
                leading_few_words = ""
            return leading_few_words + self.mask_token
        elif level == 'token':
            return ''
    
    def __call__(self, text: str, reduce_ratio: float = 0.35, reduce_level :str = 'phrase') -> List[str]:
        context = self.beautify_context(text)

        self.mask_ratio = reduce_ratio

        sents = [sent.strip() for sent in re.split(self.sent_tokenize_pattern, context) if sent.strip()]

        # You want the reduce happen at sentence level, phrase level, or token level?
        assert reduce_level in ['sent', 'phrase', 'token'], f"reduce_level should be one of ['sent', 'phrase', 'token'], got {reduce_level}"
        sent_lus, phrase_lus, token_lus = self._lexical_unit(sents)
        lexical_level = {
            'sent': sent_lus,
            'phrase': phrase_lus,
            'token': token_lus
        }

        # context is the reduced context, masked_sents denotes what context has been filtered out
        context, masked_sents = self.self_info_mask(lexical_level[reduce_level].text, lexical_level[reduce_level].self_info, reduce_level)
        return context, masked_sents

def main(
    model_name = 'TheBloke/Llama-2-7B-Chat-GPTQ',
    file_to_process: str = None,
    file_to_save: str = None,
):

    sc = SelectiveContext(model_name=model_name)

    if file_to_process is None:
        while True:
            text = input("Please input the text you want to reduce: ")
            if text == 'exit':
                break
            context, masked_sents = sc(text)
            print_context_reduced_context(context, masked_sents)
    else:
        with open(file_to_process, 'r') as f:
            text = f.read()
        context, masked_sents = sc(text)
        if file_to_save is not None:
            with open(file_to_save, 'w') as f:
                f.write(context)
        else:
            print_context_reduced_context(context, masked_sents)


def print_context_reduced_context(context, masked_sents):
    print('***********\nThe resultsing context is: \n')
    print(context, '\n\n')
    print('***********\nThe content that has been filtered out is: \n')
    print(masked_sents, '\n\n')


if __name__ == "__main__":
    main(model_name='TheBloke/Llama-2-7B-Chat-GPTQ')
