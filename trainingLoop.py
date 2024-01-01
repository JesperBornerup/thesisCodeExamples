#import stuff
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from transformers import BertConfig, BertForTokenClassification
from transformers import EarlyStoppingCallback
from torch.utils.data import Dataset
import os
import pickle

#Import the datasets
#This is a training loop for the BERT base model. This is not a parameter grid search

nerTrainValAugment = pickle.load(open("nerSplitsAugmentSyn.pkl", "rb"))
nerTest = pickle.load(open("nerTestSet.pkl", "rb"))
nerTrainVal = pickle.load(open("nerSplits.pkl", "rb"))
nerTrainValAugmentMASK = pickle.load(open("nerSplitsAugmentMask.pkl", "rb"))

def restructure_dataset(splits_dict):
    new_splits_dict = {}
    for split_index, dataset in splits_dict.items():
        new_dataset = {}
        for note, annotations in dataset.items():
            NERTAGS = []
            for annotation in annotations:
                ner_tag = annotation.get('NERTAG') or annotation.get('missingNer')
                if ner_tag and ner_tag["type"] != "Handling":
                    NERTAGS.append(ner_tag)
            # Sorting NERTAGS based on their start position
            NERTAGS.sort(key=lambda x: x['start'])
            new_dataset[note] = {'NERTAGS': NERTAGS}
        new_splits_dict[split_index] = new_dataset
    return new_splits_dict

# Restructuring nerTrainVal dataset
nerTrainVal_restructuredAugment = restructure_dataset(nerTrainValAugment)
nerTrainValAugment = nerTrainVal_restructuredAugment
# Since nerTest is not split, we make it a dictionary with a single key
nerTest_dict = {0: nerTest}
nerTest_restructured = restructure_dataset(nerTest_dict)

nerTest = nerTest_restructured

nerTrainVal_restructured = restructure_dataset(nerTrainVal)
nerTrainVal = nerTrainVal_restructured

nerTrainVal_restructuredAugmentMASK = restructure_dataset(nerTrainValAugmentMASK)
nerTrainValAugmentMASK = nerTrainVal_restructuredAugmentMASK



#Create set of unique entities for the labels

unique_entity_types = set()

for split in nerTrainVal.values():
    for annotations in split.values():
        for annotation in annotations.get('NERTAGS', []):  # Assuming each note has 'NERTAGS' key
            entity_type = annotation.get('type')
            if entity_type:
                unique_entity_types.add(entity_type)
for annotations in nerTest.values():
    for annotation in annotations.get('NERTAGS', []):  # Assuming each note has 'NERTAGS' key
        entity_type = annotation.get('type')
        if entity_type:
            unique_entity_types.add(entity_type)

#define early stopping 

early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=3,  # Number of steps with no improvement after which training will be stopped
    early_stopping_threshold=0.0  # Minimum improvement to qualify as an improvement
)

#define custom dataset to label the entities

class NERDataset(Dataset):
    def __init__(self, data, tokenizer, label_to_id, max_length):
        self.data = data  # Data should be a dictionary of notes and their annotations
        self.tokenizer = tokenizer
        self.label_to_id = label_to_id  # Mapping of NER labels to numerical IDs
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = list(self.data.items())[idx]
        note, annotation = item
        if not isinstance(note, str):
            note = str(note)
            print(f"Non-string note found: {note} (type: {type(note)}")

        tokenized_inputs = self.tokenizer(note, truncation=True, padding='max_length', max_length=self.max_length, return_offsets_mapping=True)
        labels = [self.label_to_id['O']] * len(tokenized_inputs['input_ids'])

        NERTAGS = annotation.get('NERTAGS', [])
        for tag in NERTAGS:
            label = tag['type']
            start = tag['start']
            end = tag['end']

            # Ensure start and end are within the note's length
            start = min(max(start, 0), len(note) - 1)
            end = min(max(end, 0), len(note) - 1)

            start_position = tokenized_inputs.char_to_token(start)
            end_position = tokenized_inputs.char_to_token(end)

            if start_position is not None and end_position is not None:
                labels[start_position] = self.label_to_id[f'B-{label}']
                for i in range(start_position + 1, end_position + 1):
                    labels[i] = self.label_to_id[f'I-{label}']

        labels = torch.tensor(labels[:self.max_length])
        labels = torch.nn.functional.pad(labels, pad=(0, self.max_length - len(labels)), value=self.label_to_id['O'])

        return {
            'input_ids': torch.tensor(tokenized_inputs['input_ids']),
            'attention_mask': torch.tensor(tokenized_inputs['attention_mask']),
            'labels': labels
        }
    

#K is the number of splits
k=6
modelName="bert-base-cased"

#Validation loop
for idx in range(0, k):
    val_data = {}
    train_data = {}
    allData = {}
    for idData, data in nerTrainVal.items():
        if idData == idx:
            val_data = nerTrainVal[idx]
            allData.update(data)
        else:
            train_data.update(data)
            allData.update(data)
    #Create label to id mapping
    label_list = ['O'] + [f'B-{label}' for label in unique_entity_types]
    label_list += [f'I-{label}' for label in unique_entity_types]
    label_to_id = {label: id for id, label in enumerate(label_list)}

    tokenizer = AutoTokenizer.from_pretrained(modelName)

    train_dataset = NERDataset(train_data, tokenizer, label_to_id, max_length=128)
    val_dataset = NERDataset(val_data, tokenizer, label_to_id, max_length=128)

    def extract_labels(dataset):
        labels = []
        for i in range(len(dataset)):
            data = dataset[i]
            ner_tags = data['labels'].tolist()
            labels.extend([label_list[label] for label in ner_tags])
        return labels
    
    train_labels = extract_labels(train_dataset)
    train_labels_set = set(train_labels)
    filtered_label_list = [label for label in label_list if label in train_labels_set]
    #Apply weights to the classes
    filtered_class_weights = compute_class_weight('balanced', classes=np.unique(filtered_label_list), y=train_labels)
    #Classes not in train, but might appear in val. Gets max weight
    max_weight = max(filtered_class_weights)
    full_class_weights = np.ones(len(label_list)) * max_weight
    for label, weight in zip(filtered_label_list, filtered_class_weights):
        full_class_weights[label_to_id[label]] = weight
    
    weights = torch.tensor(full_class_weights, dtype=torch.float).to('cuda')

    #Compute metrics function that also logs the clasification report when called
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        
        report = classification_report(true_labels, true_predictions, zero_division=1, output_dict=True)

        pickle_file_path = 'classification_reports_bertBasedCased.pkl'

        # Check if the file exists and load it, otherwise create an empty dictionary
        if os.path.exists(pickle_file_path):
            with open(pickle_file_path, 'rb') as file:
                data = pickle.load(file)
        else:
            data = {}
        # Append new data
        idx2 = (len(data) % 45) + 1
        data_key = f'epoch:{idx2}_fold:{idx}'
        data[data_key] = report

        # Save the updated data to the pickle file
        with open(pickle_file_path, 'wb') as file:
            pickle.dump(data, file)

            return {
                "precision": precision_score(true_labels, true_predictions, average='micro', zero_division=1),
                "recall": recall_score(true_labels, true_predictions, average='micro', zero_division=1),
                "f1": f1_score(true_labels, true_predictions, average='micro', zero_division=1),
            }
    
    #Call compute metrics every epoch. 
    training_args = TrainingArguments(
        output_dir=f'./results/{modelName}_EPOCHS',
        learning_rate=3e-05,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=45,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_total_limit=5,
        )
    #Custom trainer that implements the weights
    class CustomTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # You can add any additional initializations here if needed

        def compute_loss(self, model, inputs, return_outputs=False):
            # Extract the labels from inputs
            labels = inputs.get("labels")

            # Forward pass
            outputs = model(**inputs)
            logits = outputs.get('logits')

            # Compute custom loss
            loss_fct = torch.nn.CrossEntropyLoss(weight=weights)
            active_loss = labels.view(-1) != -100
            active_logits = logits.view(-1, model.num_labels)
            active_labels = torch.where(
                active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
            )

            loss = loss_fct(active_logits, active_labels)

            return (loss, outputs) if return_outputs else loss
        
    model = AutoModelForTokenClassification.from_pretrained(modelName, num_labels=len(label_list)).to('cuda')
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()



