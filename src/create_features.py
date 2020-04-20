import ast
import errno
import os
import pickle
import warnings

import numpy as np
import pandas as pd
import spacy
from nltk.stem.lancaster import LancasterStemmer
from scipy import spatial
from textblob import TextBlob

warnings.filterwarnings('ignore')
#
# embeddings_paths = [
#     'data/full_data/dict_embeddings1_fast_text.pickle',
#     'data/full_data/dict_embeddings2_fast_text.pickle'
# ]
# squad_preprocessed_data_path = "data/train-v2.0.csv"
# output_csv_path = "data/train-v2.0_detect_sent_fast_text.csv"
# output_csv_with_root_matching_path = "data/train-v2.0_detect_sent_root_matching_fast_text.csv"
# root_matching = True  # Takes a long time

embeddings_paths = [
    'data/train2.0_embeddings1.pickle',
    'data/train2.0_embeddings2.pickle'
]
squad_preprocessed_data_path = "data/train2.0.csv"
output_csv_path = "data/train2.0_detect_sent.csv"
output_csv_with_root_matching_path = "data/train2.0_detect_sent_root_matching.csv"
root_matching = True  # Takes a long time


def silentremove(filename):
    try:
        os.remove(filename)
    except OSError as e:
        if e.errno != errno.ENOENT:  # errno.ENOENT = no such file or directory
            raise  # re-raise exception if a different error occurred


# Loads embeddings into memory
def load_embeddings_as_dict(paths):
    dict_emb = {}
    for emb_path in paths:
        try:
            with open(emb_path, "rb") as f:
                d = pickle.load(f)
            dict_emb.update(d)
            del d
        except FileNotFoundError:
            print("Please check embeddings path: ", emb_path)
            raise
    return dict_emb


# Finds sentence index (from context) in which answer can be found, if does not exists return -1
def get_target_index(x):
    idx = -1
    for i in range(len(x["sentences"])):
        if x["text"] in x["sentences"][i]:
            idx = i
    return idx


# Processes training data
def process_data(df, emb_dict):
    df.dropna(inplace=True)
    # Breaking context paragraph into list of strings (sentences)
    print("- Processing sentences")
    df['sentences'] = df['context'].apply(lambda x: [item.raw for item in TextBlob(x).sentences])

    # Finds index of sentence from the context in which the answer is found
    print("- Processing targets")
    df["target"] = df.apply(get_target_index, axis=1)

    # Imports sentence embeddings into dataframe if exists
    print("- Processing sentence embeddings")
    df['sent_emb'] = df['sentences'].apply(lambda x: [emb_dict[item][0] if item in \
                                                                           emb_dict else np.zeros(4096) for item in x])
    # Imports question embeddings into dataframe if exists
    print("- Processing question embeddings")
    df['quest_emb'] = df['question'].apply(
        lambda x: emb_dict[x] if x in emb_dict else np.zeros(4096)
    )

    return df


# Computes cosine similarity distance for each question
def cosine_sim(x):
    sentence_embeddings = x["sent_emb"]

    # Take zero-th index as shape is of (1, 4096) so we simplify matrix to list
    question_embeddings = x["quest_emb"][0]

    distances = []
    for sentence_embedding in sentence_embeddings:
        distances.append(spatial.distance.cosine(sentence_embedding, question_embeddings))

    return distances


def predict_index(distances):
    if len(distances) == 0:
        raise Exception

    isnan = np.isnan(distances)

    if all(isnan):
        return 0

    return np.nanargmin(distances)


# Computes prediction from training data
def predictions(df):
    # Computes cosine similarity distance for each question
    df["cosine_sim"] = df.apply(cosine_sim, axis=1)

    # Computes euclidean distance for each question
    df["diff"] = (df["quest_emb"] - df["sent_emb"]) ** 2
    df["euclidean_dis"] = df["diff"].apply(lambda x: list(np.sum(x, axis=1)))
    del df["diff"]

    # Computes predicted index by taking the minimum distance of respective calculation
    df["cos_predicted_index"] = df["cosine_sim"].apply(lambda x: predict_index(x))
    df["euc_predicted_index"] = df["euclidean_dis"].apply(lambda x: predict_index(x))

    return df


# Computes accuracy between target and predicted
def accuracy(target, predicted):
    acc = (target == predicted).sum() / len(target)
    return acc


# ???
def match_roots(x):
    if (COUNT % 100 == 0):
        print(COUNT)

    increment()
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


def increment():
    global COUNT
    COUNT = COUNT + 1


def print_results(predicted):
    print("Prediction results: ")
    # print("cosine_sim: ", predicted["cosine_sim"][0])
    # print("euclidean_dis: ", predicted["euclidean_dis"][0])

    # Accuracy for euclidean Distance
    print("Accuracy for  euclidean Distance", accuracy(predicted["target"], predicted["euc_predicted_index"]))

    # Accuracy for Cosine Similarity
    print("Accuracy for Cosine Similarity", accuracy(predicted["target"], predicted["cos_predicted_index"]))


if __name__ == '__main__':
    # Load Embedding dictionary
    print("Loading embeddings...")
    emb_dict = load_embeddings_as_dict(embeddings_paths)

    # Load training data
    print("Loading training data...")

    if not os.path.isfile(squad_preprocessed_data_path):
        print("Please check SQUAD training data path: ", squad_preprocessed_data_path)
        raise Exception

    silentremove(output_csv_path)

    train = pd.read_csv(squad_preprocessed_data_path)
    # pre-process data
    print("Processing data")
    train = process_data(train, emb_dict)

    # Predicted Cosine & Euclidean Index
    print("Calculating predictions")
    predicted = predictions(train)
    print_results(predicted)

    # TODO: Delete unnecessary columns before saving.
    print("Saving predicted results")
    silentremove(output_csv_path)
    predicted.to_csv(output_csv_path, index=None)
    print("Saved to: ", output_csv_path)

    # Root Matching
    if root_matching:
        print("Root Matching...")

        COUNT = 0
        predicted = pd.read_csv(output_csv_path).reset_index(drop=True)
        print(predicted.shape)
        predicted["root_match_idx"] = predicted.apply(match_roots, axis=1)
        predicted["root_match_idx_first"] = predicted["root_match_idx"].apply(lambda x: x[0] if len(x) > 0 else 0)
        silentremove(output_csv_with_root_matching_path)
        predicted.to_csv(output_csv_with_root_matching_path, index=None)
