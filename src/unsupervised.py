import numpy as np
import pandas as pd
import ast
from textblob import TextBlob
import pickle
from scipy import spatial
import spacy
from nltk.stem.lancaster import LancasterStemmer

import warnings

warnings.filterwarnings('ignore')


# Loads embeddings into memory
def load_embeddings(embeddings_path):
    dict_emb = {}
    for emb_path in embeddings_path:
        with open(emb_path, "rb") as f:
            d = pickle.load(f)

        dict_emb.update(d)
        del d

    return dict_emb


# Finds sentence index (from context) in which answer can be found, if does not exists return -1
def get_target_index(x):
    idx = -1
    for i in range(len(x["sentences"])):
        if x["text"] in x["sentences"][i]:
            idx = i
    return idx


# Processes training data
def process_data(train, embeddings):
    # Breaking context paragraph into list of strings (sentences)
    print("- Processing sentences")
    train['sentences'] = train['context'].apply(lambda x: [item.raw for item in TextBlob(x).sentences])

    # Finds index of sentence from the context in which the answer is found
    print("- Processing targets")
    train["target"] = train.apply(get_target_index, axis=1)

    # Imports sentence embeddings into dataframe if exists
    print("- Processing sentence embeddings")
    train['sent_emb'] = train['sentences'].apply(
        lambda x: [embeddings[item][0] if item in embeddings else np.zeros(4096) for item in x])

    # Imports question embeddings into dataframe if exists
    print("- Processing question embeddings")
    train['quest_emb'] = train['question'].apply(lambda x: embeddings[x] if x in embeddings else np.zeros(4096))

    return train


# Computes cosine similarity distance for each question
def cosine_sim(x):
    sentence_embeddings = x["sent_emb"]

    # Take zero-th index as shape is of (1, 4096) so we simplify matrix to list
    question_embeddings = x["quest_emb"][0]

    distances = []
    for sentence_embedding in sentence_embeddings:
        distances.append(spatial.distance.cosine(sentence_embedding, question_embeddings))
    return distances


# Computes prediction from training data
def predictions(train):
    # Computes cosine similarity distance for each question
    train["cosine_sim"] = train.apply(cosine_sim, axis=1)

    # Computes euclidean distance for each question
    train["diff"] = (train["quest_emb"] - train["sent_emb"]) ** 2
    train["euclidean_dis"] = train["diff"].apply(lambda x: list(np.sum(x, axis=1)))
    del train["diff"]

    # Computes predicted index by taking the minimum distance of respective calculation
    train["cos_predicted_index"] = train["cosine_sim"].apply(lambda x: np.argmin(x))
    train["euc_predicted_index"] = train["euclidean_dis"].apply(lambda x: np.argmin(x))

    return train


# Computes accuracy between target and predicted
def accuracy(target, predicted):
    acc = (target == predicted).sum() / len(target)
    return acc


# ???
def match_roots(x):
    # Create nltk stemmer
    stemmer = LancasterStemmer()

    # Create spacy language processor
    nlp_processor = spacy.load('en')

    question = x["question"].lower()
    sentences = nlp_processor(x["context"].lower()).sents

    # Calculates the stemmed root word of the question
    question_root = stemmer.stem(str([sent.root for sent in nlp_processor(question).sents][0]))

    matched = []
    for i, sent in enumerate(sentences):
        # List of roots from sentence surrounding nouns
        roots = [stemmer.stem(chunk.root.head.text.lower()) for chunk in sent.noun_chunks]

        # If question root exists in sentence roots (surrounding nouns)
        if question_root in roots:
            for k, item in enumerate(ast.literal_eval(x["sentences"])):
                # If sentence is found in match
                if str(sent) in item.lower():
                    matched.append(k)

    return matched


def print_results(predicted):
    print("Prediction results: ")
    print("cosine_sim: ", predicted["cosine_sim"][0])
    print("euclidean_dis: ", predicted["euclidean_dis"][0])

    # Accuracy for euclidean Distance
    print("Accuracy for  euclidean Distance", accuracy(predicted["target"], predicted["euc_predicted_index"]))

    # Accuracy for Cosine Similarity
    print("Accuracy for Cosine Similarity", accuracy(predicted["target"], predicted["cos_predicted_index"]))


embeddings_paths = ['data/full_data/dict_embeddings1.pickle', 'data/full_data/dict_embeddings2.pickle']
data_path = "data/train.csv"

root_matching = True  # Takes a long time
save_data = True

# Load Embedding dictionary
print("Loading embeddings...")
emb_dict = load_embeddings(embeddings_paths)

# Load traning data
print("Loading training data...")
train = pd.read_csv(data_path)

# Process data
print("Processing data...")
train.dropna(inplace=True)
train = process_data(train, emb_dict)

# Predicted Cosine & Euclidean Index
print("Calculating predictions...")
predicted = predictions(train)
print_results(predicted)

if save_data:
    print("Saving predicted results...")
    predicted.to_csv("train_detect_sent_final.csv", index=None)
    print("Saved.")

# Root Matching
if root_matching:
    print("Root Matching...")

    predicted = pd.read_csv("train_detect_sent_final.csv").reset_index(drop=True)
    predicted["root_match_idx"] = predicted.apply(match_roots, axis=1)
    predicted["root_match_idx_first"] = predicted["root_match_idx"].apply(lambda x: x[0] if len(x) > 0 else 0)
    predicted.to_csv("train_detect_sent.csv", index=None)
