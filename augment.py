from gensim.models import KeyedVectors
import random

MODEL_FILE = 'dsl_skipgram_2020_m5_f500_epoch2_w5.model.w2v.bin'
modelG = KeyedVectors.load_word2vec_format(MODEL_FILE, binary=True)


#AugmentSYN

import re
import random
from collections import Counter
import copy

def augment_dataset_with_synonyms(data, model, listOfTypes, max_attempts=6):
    augmented_data = {}
    original_data = copy.deepcopy(data)

    for note, annotations in original_data.items():
        # Extract entities and decide whether to process this note
        entities = []
        process_note = False
        for annotation in annotations:
            ner_tag = annotation.get("NERTAG") or annotation.get("missingNer")
            if ner_tag:
                entities.append([ner_tag['start'], ner_tag['end'], ner_tag['entity']])
                if not listOfTypes or ner_tag['type'] in listOfTypes:
                    process_note = True
        augmented_note = note
        if process_note:
            words = re.findall(r'\b\w+\b', note)
            word_counts = Counter(words)
            entity_texts = [note[start:end+1] for start, end, _ in entities]

            # Exclude words that are part of or substrings of entities
            replaceable_words = [word for word in words if all(word not in entity and not entity.startswith(word) for entity in entity_texts) and word_counts[word] == 1]
            attempts = 0
            while attempts < max_attempts and replaceable_words:
                word_to_replace = random.choice(replaceable_words)
                if word_to_replace in model.key_to_index:
                    similar_word = model.most_similar(positive=[word_to_replace], topn=1)[0][0]
                    
                    # Find the index of the word to be replaced
                    replace_index = augmented_note.find(word_to_replace)
                    
                    # Perform the replacement
                    augmented_note = note[:replace_index] + augmented_note[replace_index:].replace(word_to_replace, similar_word, 1)

                    offset = len(similar_word) - len(word_to_replace)

                    for annotation in annotations:
                        ner_tag = annotation.get("NERTAG") or annotation.get("missingNer")
                        if ner_tag:
                            # Adjust indices based on the replace_index
                            if ner_tag['start'] >= replace_index:
                                ner_tag['start'] += offset
                                ner_tag['end'] += offset
                                # Update the annotation
                                if "NERTAG" in annotation:
                                    annotation["NERTAG"] = ner_tag
                                else:
                                    annotation["missingNer"] = ner_tag
                    break
                else:
                    replaceable_words.remove(word_to_replace)
                    attempts += 1

        # Update the augmented data
        augmented_data[augmented_note] = annotations
    return augmented_data

def check_annotation_alignment(augmented_data):
    for note, annotations in augmented_data.items():
        for annotation in annotations:
            ner_tag = annotation.get('NERTAG', annotation.get('missingNer'))
            if ner_tag:
                entity = ner_tag['entity']
                start, end = ner_tag['start'], ner_tag['end']
                substring = note[start:end+1]

                if entity != substring:
                    print(len(entity)-len(substring))
                    print(note)
                    print(f"Entity: {entity}")
                    print(f"Substring: {substring}")
                    print(f"Type: {ner_tag['type']}")
                    print("Misalignment detected\n")


#AugmentMASK
                    
from transformers import BertTokenizer, BertForMaskedLM
import torch
import random
import re
import copy
import string

# Load Danish BERT model
MODEL_NAME = "Maltehb/danish-bert-botxo"
model = BertForMaskedLM.from_pretrained(MODEL_NAME).to('cuda')
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

def is_single_non_digit_word(word):
    return len(re.findall(r'\b\w+\b', word)) == 1 and not any(char.isdigit() for char in word)

def mask_and_predict(note, word_to_mask, model, tokenizer):
    masked_note = note.replace(word_to_mask, tokenizer.mask_token, 1)
    input_ids = tokenizer.encode(masked_note, return_tensors='pt').to('cuda')
    
    with torch.no_grad():
        logits = model(input_ids).logits

    mask_token_index = torch.where(input_ids == tokenizer.mask_token_id)[1]
    mask_token_logits = logits[0, mask_token_index, :]
    top_token = torch.argmax(mask_token_logits, dim=1)

    predicted_token = tokenizer.decode(top_token.cpu(), skip_special_tokens=True).strip()
    return predicted_token

def augment_dataset_with_bert(data, model, tokenizer, listOfTypes=None, max_attempts=6):
    augmented_data = {}
    original_data = copy.deepcopy(data)

    for note, annotations in original_data.items():
        # Extract entities and decide whether to process this note
        entities = []
        process_note = False
        for annotation in annotations:
            ner_tag = annotation.get("NERTAG") or annotation.get("missingNer")
            if ner_tag:
                entities.append([ner_tag['start'], ner_tag['end'], ner_tag['entity']])
                if not listOfTypes or ner_tag['type'] in listOfTypes:
                    process_note = True
        augmented_note = note
        if process_note:
            words = re.findall(r'\b\w+\b', note)
            word_counts = Counter(words)
            entity_texts = [note[start:end+1] for start, end, _ in entities]

            # Exclude words that are part of or substrings of entities
            replaceable_words = [word for word in words if all(word not in entity and not entity.startswith(word) for entity in entity_texts) and word_counts[word] == 1]
            attempts = 0
            while attempts < max_attempts and replaceable_words:
                word_to_replace = random.choice(replaceable_words)
                replace_index = augmented_note.find(word_to_replace)
                predicted_word = mask_and_predict(augmented_note, word_to_replace, model, tokenizer)
                offset = len(predicted_word) - len(word_to_replace)
                if predicted_word and is_single_non_digit_word(predicted_word):
                    augmented_note = note[:replace_index] + augmented_note[replace_index:].replace(word_to_replace, predicted_word, 1)
                    for annotation in annotations:
                        ner_tag = annotation.get("NERTAG") or annotation.get("missingNer")
                        if ner_tag and ner_tag['start'] >= note.find(word_to_replace):
                            ner_tag['start'] += offset
                            ner_tag['end'] += offset
                            if "NERTAG" in annotation:
                                annotation["NERTAG"] = ner_tag
                            else:
                                annotation["missingNer"] = ner_tag
                    break
                else:
                    attempts += 1
                    replaceable_words.remove(word_to_replace)

        augmented_data[augmented_note] = annotations

    return augmented_data
