import warnings

from InferSent.models import InferSent

warnings.filterwarnings('ignore')
import pickle
import pandas as pd
from textblob import TextBlob
import torch
import spacy

en_nlp = spacy.load('en')

# CONFIG
embeddings_paths = [
    'data/full_data/dict_embeddings1_fast_text.pickle',
    'data/full_data/dict_embeddings2_fast_text.pickle'
]
squad_dataset_path = "../squad/train-v2.0.json"
output_dataset_as_csv_path = "data/train-v2.0.csv"

infersent_pretrained_path = 'InferSent/encoder/infersent2.pkl'
glove_path = "InferSent/dataset/fastText/crawl-300d-2M.vec"

full_data = True


def populate_dataframe(training):
    contexts = []
    questions = []
    answers_text = []
    answers_start = []

    _size = training.shape[0] if full_data else 2

    for i in range(_size):
        topic = training['data'][i]['paragraphs']
        for sub_para in topic:
            for q_a in sub_para['qas']:
                questions.append(q_a['question'])
                # Squad 2.0 - answers may be empty.
                if len(q_a['answers']) == 0:
                    answers_start.append(-1)
                    answers_text.append(None)
                else:
                    answers_start.append(q_a['answers'][0]['answer_start'])
                    answers_text.append(q_a['answers'][0]['text'])
                contexts.append(sub_para['context'])
    df = pd.DataFrame({"context": contexts, "question": questions, "answer_start": answers_start, "text": answers_text})
    return df


def generate_embeddings(df):
    paras = list(df["context"].drop_duplicates().reset_index(drop=True))

    print("Paragraph count:", len(paras))

    blob = TextBlob(" ".join(paras))
    sentences = [item.raw for item in blob.sentences]

    params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                    'pool_type': 'max', 'dpout_model': 0.0, 'version': 2}
    infersent = InferSent(params_model)
    infersent.load_state_dict(torch.load(infersent_pretrained_path))
    infersent.set_w2v_path(glove_path)

    print("Building Infersent vocabulary")
    infersent.build_vocab(sentences, tokenize=True)

    dict_embeddings = {}

    print("Building sentence embeddings")
    print("Sentence count:", len(sentences))
    for i in range(len(sentences)):
        dict_embeddings[sentences[i]] = infersent.encode([sentences[i]], tokenize=True)

    print("Building question embeddings")
    questions = df["question"].tolist()
    print("Questions count:", len(questions))
    for i in range(len(questions)):
        dict_embeddings[questions[i]] = infersent.encode([questions[i]], tokenize=True)

    return dict_embeddings


if __name__ == '__main__':
    train = pd.read_json(squad_dataset_path)

    dataframe = populate_dataframe(train)
    dataframe.to_csv(output_dataset_as_csv_path, index=False)
    print("Saved data to: ", output_dataset_as_csv_path)

    print("Creating embeddings")
    dict_embeddings = generate_embeddings(dataframe)

    for index, embeddings_path in enumerate(embeddings_paths):
        with open(embeddings_path, 'wb') as handle:
            embedding_half = {key: dict_embeddings[key] for i, key in enumerate(dict_embeddings) if
                              i % len(embeddings_paths) == index}
            pickle.dump(embedding_half, handle)
            print("Saved: ", embeddings_path)
