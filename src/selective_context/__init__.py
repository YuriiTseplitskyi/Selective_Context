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

    def __init__(
        self, 
        model_name: str = 'TheBloke/Llama-2-7B-Chat-GPTQ', 
        model_config: dict = {},
        device: str = "cuda"
    ):

        self.model_name = model_name
        self.device = device
        self.model_config = model_config

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
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
            
            # Some models require legacy padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **self.model_config)
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
            
            # Handle empty input case
            if encoding['input_ids'].shape[1] == 0:
                return [], []
            
            # Get model outputs
            outputs = self.model(**encoding)
            logits = outputs.logits
            
            # Calculate probabilities and self-information
            probs = torch.softmax(logits, dim=-1)
            self_info = -torch.log(probs)
            
            # Get input IDs for token decoding
            input_ids = encoding['input_ids']
            
            # Handle the case where we might have only a single token
            if input_ids.shape[1] <= 1:
                # If we only have the prefix token, return empty lists
                if prefixed_text != text:
                    return [], []
                # Otherwise decode the single token
                token = self.tokenizer.decode(input_ids[0][0].item())
                info_val = self_info[0, 0, input_ids[0][0].item()].item()
                return [token], [info_val]
            
            # Skip the prefix token in the result if we added one
            skip_tokens = 1 if prefixed_text != text else 0
            
            # Decode tokens and gather self-information
            tokens = []
            token_self_info = []
            
            # Process each token individually
            for i in range(skip_tokens, input_ids.shape[1]-1):
                token_id = input_ids[0, i].item()
                next_token_id = input_ids[0, i+1].item()
                
                # Decode the token
                token = self.tokenizer.decode([token_id], skip_special_tokens=False)
                tokens.append(token)
                
                # Get self-information for the next token
                info_val = self_info[0, i, next_token_id].item()
                token_self_info.append(info_val)
            
            # Add the last token if we're not skipping it (it will have no next token prediction)
            if input_ids.shape[1] > skip_tokens:
                last_token_id = input_ids[0, -1].item()
                tokens.append(self.tokenizer.decode([last_token_id], skip_special_tokens=False))
                # Use the average info value for the last token as there's no next token
                if token_self_info:
                    token_self_info.append(sum(token_self_info) / len(token_self_info))
                else:
                    token_self_info.append(0.0)
            
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
                try:
                    # Get tokens and self-information
                    tokens, self_info = self.get_self_information(sent)
                    
                    # Handle empty self_info arrays
                    if not self_info:
                        sent_self_info.append(0.0)  # Default value
                    else:
                        sent_self_info.append(np.mean(self_info))

                    all_tokens.extend(tokens)
                    all_token_self_info.extend(self_info)

                    # Calculate lexical units
                    noun_phrases, noun_phrases_info = self._calculate_lexical_unit(tokens, self_info)

                    # We need to add a space before the first noun phrase for every sentence except the first one
                    if noun_phrases and all_noun_phrases:
                        noun_phrases[0] = f" {noun_phrases[0]}"
                        
                    all_noun_phrases.extend(noun_phrases)
                    all_noun_phrases_info.extend(noun_phrases_info)
                    
                except Exception as e:
                    print(f"Warning: Error processing sentence: '{sent[:50]}...'. Error: {e}")
                    # Add default values if processing fails
                    sent_self_info.append(0.0)
            
            # Ensure we have at least one item in each list
            if not all_noun_phrases:
                all_noun_phrases = [""]
                all_noun_phrases_info = [0.0]
                
            if not all_tokens:
                all_tokens = [""]
                all_token_self_info = [0.0]
            
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
            
            # Handle empty lists by providing a default value
            unit_self_info_ = []
            for info_list in unit_self_info:
                if not info_list:  # If the list is empty
                    unit_self_info_.append(0.0)  # Use default value of 0.0
                else:
                    unit_self_info_.append(np.mean(info_list))
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
            
            try:
                # Handle case where no noun phrases are found
                noun_phrases = _noun_phrases(sent)
                if not noun_phrases:
                    # If no noun phrases, use the whole sentence as a single phrase
                    noun_phrases = [sent]
                    
                noun_phrases_info = _unit_info(tokens, self_info, noun_phrases)
                return noun_phrases, noun_phrases_info
                
            except Exception as e:
                print(f"Warning: Error processing noun phrases: {e}")
                # Return a fallback value
                return [sent], [np.mean(self_info) if self_info else 0.0]
        
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
        
        # Handle empty inputs
        if not sents or not self_info:
            return "", []
        
        # Filter out NaN and None values from self_info
        valid_info = [info for info in self_info if info is not None and not np.isnan(info)]
        
        # If no valid values, return original text
        if not valid_info:
            self.ppl_threshold = 0.0
        else:
            self.ppl_threshold = np.nanpercentile(self_info, self.mask_ratio * 100)

        for sent, info in zip(sents, self_info):
            # Check for NaN or None values
            if info is None or np.isnan(info):
                sents_after_mask.append(sent)  # Keep the original text
            elif info < self.ppl_threshold:
                masked_sents.append(sent)
                sents_after_mask.append(self.mask_a_sent(sent, mask_level))
            else:
                sents_after_mask.append(sent)
        
        # Join with proper spacing based on level
        if mask_level == 'sent':
            masked_context = " ".join(sents_after_mask)
        else:
            # For phrase and token level, be careful with spacing
            masked_context = ""
            for i, s in enumerate(sents_after_mask):
                if i > 0 and s and not s.startswith(" ") and not masked_context.endswith(" "):
                    masked_context += " " + s
                else:
                    masked_context += s
        
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
        """
        Process text and reduce context based on specified level and ratio.
        
        Args:
            text (str): Input text to process
            reduce_ratio (float): Percentage of content to reduce (0.0 to 1.0)
            reduce_level (str): Level at which to apply reduction ('sent', 'phrase', or 'token')
        
        Returns:
            Tuple[str, List[str]]: Reduced context and masked content
        """
        context = self.beautify_context(text)
        self.mask_ratio = reduce_ratio

        # First, ensure we preserve paragraph breaks
        paragraphs = context.split("\n\n")
        all_results = []
        all_masked = []
        
        for para in paragraphs:
            if not para.strip():
                continue
                
            # Split into sentences
            sents = [sent.strip() for sent in re.split(self.sent_tokenize_pattern, para) if sent.strip()]
            
            # You want the reduce happen at sentence level, phrase level, or token level?
            assert reduce_level in ['sent', 'phrase', 'token'], f"reduce_level should be one of ['sent', 'phrase', 'token'], got {reduce_level}"
            
            try:
                sent_lus, phrase_lus, token_lus = self._lexical_unit(sents)
                
                lexical_level = {
                    'sent': sent_lus,
                    'phrase': phrase_lus,
                    'token': token_lus
                }
                
                # context is the reduced context, masked_sents denotes what context has been filtered out
                reduced_para, masked_sents = self.self_info_mask(
                    lexical_level[reduce_level].text, 
                    lexical_level[reduce_level].self_info, 
                    reduce_level
                )
                
                all_results.append(reduced_para)
                all_masked.extend(masked_sents)
                
            except Exception as e:
                print(f"Error processing paragraph: {e}")
                all_results.append(para)  # Fall back to original paragraph
        
        # Join paragraphs with appropriate spacing
        final_context = "\n\n".join(all_results)
        
        return final_context, all_masked

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
