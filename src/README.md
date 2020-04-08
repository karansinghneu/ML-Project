# CS 6140 Term Project

## Team Members
- Abhinav Agrawal
- Karan Singh
- Shawn Martin

### Building the QA system for [Stanford Question Answering Datatset](https://rajpurkar.github.io/SQuAD-explorer/)

### Files
- `create_embeddings.py`:  Creates a dictionary of sentence embeddings for all the sentences and questions in the wikipedia articles of training dataset

- `unsupervised.py`:  Calculates the distance between sentence & questions basis Euclidean & Cosine similarity using sentence embeddings. It finally extracts the setence from each paragraph that has the minimum distance from the question. Currently, they are giving an accuracy of 45% & 63% respectively.

- `supervised.py`: Treats this problem as supervised learning problem where I am fitting multinomial logistic regression, random forest and xgboost and create 20 features - (2 features represnts the cosine distance & euclidean for one sentence. I am limiting each para to 10 sentences). The target variable is the sentence ID having the correct answer. So I have 10 labels. This is currently giving an accuracy of 63%, 65% & 69% respectively.

### Running instructions

- `cd src`
- `python3 -m venv env && source env/bin/activate`
- `python3 -m spacy download en`
- `python3 -m textblob.download_corpora`
- `mkdir encoder && curl -Lo encoder/infersent1.pkl https://dl.fbaipublicfiles.com/infersent/infersent1.pkl && curl -Lo encoder/infersent2.pkl https://dl.fbaipublicfiles.com/infersent/infersent2.pkl`
- `./InferSent/dataset/get_data.bash`

