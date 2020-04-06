import warnings

from Infersent.models import InferSent

warnings.filterwarnings('ignore')
import pickle
import pandas as pd
from textblob import TextBlob
import torch
import spacy

en_nlp = spacy.load('en')

if __name__ == '__main__':

    ## Convert Json to Pandas Dataframe

    train = pd.read_json("data/train-v1.1.json")

    valid = pd.read_json("data/dev-v1.1.json")

    print("Training set: ", train.shape)
    train.head(3)
    print(train.iloc[1, 0]['paragraphs'][0])

    print("Validation set: ", valid.shape)
    valid.head(3)
    # valid.iloc[1,0]['paragraphs'][0]

    contexts = []
    questions = []
    answers_text = []
    answers_start = []

    for i in range(train.shape[0]):
        topic = train.iloc[i, 0]['paragraphs']
        for sub_para in topic:
            for q_a in sub_para['qas']:
                questions.append(q_a['question'])
                answers_start.append(q_a['answers'][0]['answer_start'])
                answers_text.append(q_a['answers'][0]['text'])
                contexts.append(sub_para['context'])
    df = pd.DataFrame({"context": contexts, "question": questions, "answer_start": answers_start, "text": answers_text})

    df.to_csv("data/train.csv", index=None)

    paras = list(df["context"].drop_duplicates().reset_index(drop=True))

    blob = TextBlob(" ".join(paras))
    sentences = [item.raw for item in blob.sentences]

    print("Sentence count:", len(sentences))

    params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                    'pool_type': 'max', 'dpout_model': 0.0, 'version': 1}
    infersent = InferSent(params_model)
    infersent.load_state_dict(torch.load('Infersent/encoder/infersent1.pkl'))
    infersent.set_w2v_path("Infersent/dataset/GloVe/glove.840B.300d.txt")

    infersent.build_vocab(sentences, tokenize=True)

    dict_embeddings = {}

    print("Building sentences dict")
    # for i in range(len(sentences)):
    for i in range(1000):
        if i % 100 == 0:
            print(i)
        dict_embeddings[sentences[i]] = infersent.encode([sentences[i]], tokenize=True)

    questions = list(df["question"])

    print("Building questions dict")
    # for i in range(len(questions)):
    for i in range(1000):
        if i % 100 == 0:
            print(i)
        dict_embeddings[questions[i]] = infersent.encode([questions[i]], tokenize=True)

    d1 = {key: dict_embeddings[key] for i, key in enumerate(dict_embeddings) if i % 2 == 0}
    d2 = {key: dict_embeddings[key] for i, key in enumerate(dict_embeddings) if i % 2 == 1}

    with open('data/dict_embeddings1.pickle', 'wb') as handle:
        pickle.dump(d1, handle)
        print("Created 'data/dict_embeddings1.pickl")

    with open('data/dict_embeddings2.pickle', 'wb') as handle:
        pickle.dump(d2, handle)
        print("Created 'data/dict_embeddings2.pickl")

    del dict_embeddings
