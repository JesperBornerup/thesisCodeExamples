import pickle
import sklearn_crfsuite
from seqeval.metrics import classification_report as seqeval_report
from seqeval.metrics import f1_score as seqeval_f1_score
from gensim.models import KeyedVectors
import numpy as np

#Load datasets
data = pickle.load(open('nerSplits.pkl', 'rb'))
augmentSyn = pickle.load(open('nerSplitsAugmentSyn.pkl', 'rb'))
augmentMASK = pickle.load(open('nerSplitsAugmentMask.pkl', 'rb'))




#Load word2vector model
MODEL_FILE = 'dsl_skipgram_2020_m5_f500_epoch2_w5.model.w2v.bin'
modelG = KeyedVectors.load_word2vec_format(MODEL_FILE, binary=True)


#Example run, where split 1 is validation and the rest is training           
train = {}
validation = {}

for split, splitData in data.items():
    if split == 1:
        validation = splitData
    else:
        train.update(splitData)

def tokenize(sentence):
    return sentence.split()

def preprocess_for_crf(data):
    formatted_data = []

    for sentence, annotations in data.items():
        tokens = tokenize(sentence)
        labels = ["O"] * len(tokens)  # Initialize labels as 'Outside'

        for annotation in annotations:
            ner_tag = annotation.get('NERTAG') or annotation.get('missingNer')
            if ner_tag:
                start = ner_tag["start"]
                end = ner_tag["end"]
                entity_type = ner_tag["type"]

                # Find the token indices for start and end
                start_idx = len(tokenize(sentence[:start]))
                end_idx = start_idx + len(tokenize(sentence[start:end]))

                if start_idx < len(tokens):
                    labels[start_idx] = "B-" + entity_type
                    for i in range(start_idx + 1, min(end_idx, len(tokens))):
                        labels[i] = "I-" + entity_type

        # Combine tokens with labels
        formatted_data.extend(zip(tokens, labels))
        formatted_data.append(("", ""))  # Blank line to separate sentences

    return formatted_data

pTrain = preprocess_for_crf(train)
pVal = preprocess_for_crf(validation)

def word2features(sent, i, word2vec_model=modelG, w2v_size=modelG.vector_size):
    word = sent[i][0]
    w2v_feature = word2vec_model[word] if word in word2vec_model else np.zeros(w2v_size)

    # Features for the current word
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
    }

    # Add word2vec features for the current word
    for j in range(w2v_size):
        features[f'word2vec_{j}'] = w2v_feature[j]

    # Features for the previous word
    if i > 0:
        prev_word = sent[i-1][0]
        prev_w2v_feature = word2vec_model[prev_word] if prev_word in word2vec_model else np.zeros(w2v_size)
        features.update({
            '-1:word.lower()': prev_word.lower(),
            '-1:word.isupper()': prev_word.isupper(),
            '-1:word.istitle()': prev_word.istitle(),
            '-1:word.isdigit()': prev_word.isdigit(),
        })
        for j in range(w2v_size):
            features[f'-1:word2vec_{j}'] = prev_w2v_feature[j]
    else:
        features['BOS'] = True

    # Features for the next word
    if i < len(sent) - 1:
        next_word = sent[i+1][0]
        next_w2v_feature = word2vec_model[next_word] if next_word in word2vec_model else np.zeros(w2v_size)
        features.update({
            '+1:word.lower()': next_word.lower(),
            '+1:word.isupper()': next_word.isupper(),
            '+1:word.istitle()': next_word.istitle(),
            '+1:word.isdigit()': next_word.isdigit(),
        })
        for j in range(w2v_size):
            features[f'+1:word2vec_{j}'] = next_w2v_feature[j]
    else:
        features['EOS'] = True

    return features

# Function to transform a sentence into a sequence of features
def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, label in sent]

def split_into_sentences(preprocessed_data):
    sentences = []
    sentence = []
    for token, label in preprocessed_data:
        if token:
            sentence.append((token, label))
        else:
            sentences.append(sentence)
            sentence = []
    return sentences

train_sents = split_into_sentences(pTrain)
val_sents = split_into_sentences(pVal)

# Extract features and labels for both sets
X_train = [sent2features(s) for s in train_sents]
y_train = [sent2labels(s) for s in train_sents]

X_val = [sent2features(s) for s in val_sents]
y_val = [sent2labels(s) for s in val_sents]

# Train the CRF model
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=50,
    all_possible_transitions=True
)
# Train the CRF model
crf.fit(X_train, y_train)

# Make predictions on validation set
y_pred_val = crf.predict(X_val)


print(seqeval_report(y_val, y_pred_val))



