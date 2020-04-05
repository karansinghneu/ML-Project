import numpy as np, pandas as pd
import ast
from textblob import TextBlob
import pickle
from scipy import spatial
import warnings

warnings.filterwarnings('ignore')
import spacy
from nltk import Tree

en_nlp = spacy.load('en')
from nltk.stem.lancaster import LancasterStemmer

st = LancasterStemmer()
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer

train = pd.read_csv("data/train.csv")

print(train.shape)

### Loading Embedding dictionary

with open("data/dict_embeddings1.pickle", "rb") as f:
    d1 = pickle.load(f)

with open("data/dict_embeddings2.pickle", "rb") as f:
    d2 = pickle.load(f)

dict_emb = dict(d1)
dict_emb.update(d2)

print("dict_emb: ", len(dict_emb))

del d1, d2


## Data Processing

def get_target(x):
    idx = -1
    for i in range(len(x["sentences"])):
        if x["text"] in x["sentences"][i]: idx = i
    return idx


print("Training data")
print(train.head(3))
print(train.shape)
print("Dropping na vals")
train.dropna(inplace=True)
train.shape


def process_data(train):
    print("step 1")
    train['sentences'] = train['context'].apply(lambda x: [item.raw for item in TextBlob(x).sentences])

    print("step 2")
    train["target"] = train.apply(get_target, axis=1)

    print("step 3")
    train['sent_emb'] = train['sentences'].apply(lambda x: [dict_emb[item][0] if item in \
                                                                                 dict_emb else np.zeros(4096) for item
                                                            in x])
    print("step 4")
    train['quest_emb'] = train['question'].apply(lambda x: dict_emb[x] if x in dict_emb else np.zeros(4096))

    return train


print("Process Training data")
train = process_data(train)
print(train.head(3))

## Predicted Cosine & Euclidean Index


def cosine_sim(x):
    li = []
    for item in x["sent_emb"]:
        li.append(spatial.distance.cosine(item, x["quest_emb"][0]))
    return li

def pred_idx(distances):
    return np.argmin(distances)

def predictions(train):
    train["cosine_sim"] = train.apply(cosine_sim, axis=1)
    train["diff"] = (train["quest_emb"] - train["sent_emb"]) ** 2
    train["euclidean_dis"] = train["diff"].apply(lambda x: list(np.sum(x, axis=1)))
    del train["diff"]

    print("cosine start")

    train["pred_idx_cos"] = train["cosine_sim"].apply(lambda x: pred_idx(x))
    train["pred_idx_euc"] = train["euclidean_dis"].apply(lambda x: pred_idx(x))

    return train


predicted = predictions(train)
print("Predictions")
print(predicted.head(3))
print("cosine_sim: ", predicted["cosine_sim"][0])
print("euclidean_dis: ", predicted["euclidean_dis"][0])


## Accuracy

def accuracy(target, predicted):
    acc = (target == predicted).sum() / len(target)

    return acc


### Accuracy for  euclidean Distance

print(accuracy(predicted["target"], predicted["pred_idx_euc"]))

### Accuracy for Cosine Similarity

print(accuracy(predicted["target"], predicted["pred_idx_cos"]))

# %%

predicted.to_csv("train_detect_sent.csv", index=None)


predicted.iloc[75207, :]


ct, k = 0, 0
for i in range(predicted.shape[0]):
    if predicted.iloc[i, 10] != predicted.iloc[i, 5]:
        k += 1
        if predicted.iloc[i, 11] == predicted.iloc[i, 5]:
            ct += 1

print(ct, k)

### Combining Accuracy


label = []
for i in range(predicted.shape[0]):
    if predicted.iloc[i, 10] == predicted.iloc[i, 11]:
        label.append(predicted.iloc[i, 10])
    else:
        label.append((predicted.iloc[i, 10], predicted.iloc[i, 10]))

ct = 0
for i in range(75206):
    item = predicted["target"][i]
    try:
        if label[i] == predicted["target"][i]: ct += 1
    except:
        if item in label[i]: ct += 1


print(ct / 75206)

### Root Match

predicted = pd.read_csv("train_detect_sent.csv").reset_index(drop=True)

doc = en_nlp(predicted.iloc[0, 1])

print(predicted.iloc[0, 1])
print(predicted.iloc[0, 2])


def to_nltk_tree(node):
    if node.n_lefts + node.n_rights > 0:
        return Tree(node.orth_, [to_nltk_tree(child) for child in node.children])
    else:
        return node.orth_


[to_nltk_tree(sent.root).pretty_print() for sent in en_nlp(predicted.iloc[0, 2]).sents]


[to_nltk_tree(sent.root).pretty_print() for sent in doc.sents][5]


for sent in doc.sents:
    roots = [st.stem(chunk.root.head.text.lower()) for chunk in sent.noun_chunks]
    print(roots)



def match_roots(x):
    question = x["question"].lower()
    sentences = en_nlp(x["context"].lower()).sents

    question_root = st.stem(str([sent.root for sent in en_nlp(question).sents][0]))

    li = []
    for i, sent in enumerate(sentences):
        roots = [st.stem(chunk.root.head.text.lower()) for chunk in sent.noun_chunks]

        if question_root in roots:
            for k, item in enumerate(ast.literal_eval(x["sentences"])):
                if str(sent) in item.lower():
                    li.append(k)
    return li



predicted["question"][21493]


predicted["context"][21493]


predicted["root_match_idx"] = predicted.apply(match_roots, axis=1)


predicted["root_match_idx_first"] = predicted["root_match_idx"].apply(lambda x: x[0] if len(x) > 0 else 0)

(predicted["root_match_idx_first"] == predicted["target"]).sum() / predicted.shape[0]

predicted.to_csv("train_detect_sent.csv", index=None)

predicted[
    (predicted["sentences"].apply(lambda x: len(ast.literal_eval(x))) < 11) & (predicted["root_match_idx_first"] > 10)]

len(ast.literal_eval(predicted.iloc[21493, 4]))

question = predicted["question"][21493].lower()
sentences = en_nlp(predicted["context"][21493].lower()).sents

question_root = st.stem(str([sent.root for sent in en_nlp(question).sents][0]))

li = []
for i, sent in enumerate(sentences):
    roots = [st.stem(chunk.root.head.text.lower()) for chunk in sent.noun_chunks]
    print(roots)

    if question_root in roots: li.append(i)

ast.literal_eval(predicted["sentences"][21493])


predicted["context"][21493]


en_nlp = spacy.load('en')
sentences = en_nlp(predicted["context"][21493].lower()).sents


for item in sentences:
    print(item)

TfidfVectorizer(predicted["sentences"][0], ngram_range=(1, 2))
