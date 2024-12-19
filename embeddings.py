import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle
import json

class PPMI:
    def __init__(self, doc_path, window_size=2, k=100, tokenizer_path=None):
        self.docs = self.read_file(doc_path)
        self.window_size = window_size

        # Load or create tokenizer
        if tokenizer_path:
            with open(tokenizer_path, 'rb') as f:
                self.tokenizer = pickle.load(f)
        else:
            self.tokenizer = Tokenizer(filters='!"#$%&()*+.,/:;<=>?@[\\]^_`{|}~\t\n')
            self.tokenizer.fit_on_texts(self.docs)

        self.word_index = self.tokenizer.word_index  # Word-to-index mapping
        self.vocab_size = len(self.word_index) +1  # +1 for padding

        self.tokenized_docs = self.tokenizer.texts_to_sequences(self.docs)
        self.cooccurrence_matrix, self.vocab_index = self.build_cooccurrence_matrix(self.tokenized_docs, window_size)
        self.ppmi_matrix = self.build_ppmi_matrix(self.cooccurrence_matrix)
        self.embeddings = self.generate_embeddings()
        self.embeddings_matrix = self.Apply_SVD(k)
        self.embedding_dim = k

    def read_file(self, doc_path):
        try:
            with open(doc_path, 'r') as f:
                docs = []
                for line in f:
                    parsed_line = json.loads(line.strip())
                    if "train.SRC" in parsed_line:
                        docs.append(parsed_line["train.SRC"])
                    else:
                        raise ValueError("Missing 'train.SRC' in line.")
                if not docs:
                    raise ValueError("Document file is empty.")
                return docs
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {doc_path}")
        except json.JSONDecodeError:
            raise ValueError("Error decoding JSON from file.")

    def build_cooccurrence_matrix(self, tokenized_docs, window_size=2):
        vocab = set()
        for sentence in tokenized_docs:
            vocab.update(sentence)
        vocab = list(vocab)
        vocab_index = {word: i for i, word in enumerate(vocab)}  # create a dictionary with word as key and index as value
        cooccurrence_matrix = np.zeros((len(vocab), len(vocab)))

        for sentence in tokenized_docs:
            for i, word in enumerate(sentence):
                word_idx = vocab_index[word]
                start = max(0, i - window_size)
                end = min(len(sentence), i + window_size + 1)
                for j in range(start, end):
                    if i != j:
                        context_word_idx = vocab_index[sentence[j]]
                        cooccurrence_matrix[word_idx][context_word_idx] += 1

        return cooccurrence_matrix, vocab_index

    def build_ppmi_matrix(self, cooccurrence_matrix):
        total_count = np.sum(cooccurrence_matrix)
        word_count = np.sum(cooccurrence_matrix, axis=1)
        context_count = np.sum(cooccurrence_matrix, axis=0)  # sum of each column
        ppmi_matrix = np.zeros_like(cooccurrence_matrix)  # create a zero matrix with the same shape as cooccurrence_matrix
        for i in range(cooccurrence_matrix.shape[0]):
            for j in range(cooccurrence_matrix.shape[1]):
                if cooccurrence_matrix[i][j] > 0:
                    p_ij = cooccurrence_matrix[i][j] / total_count
                    p_i = word_count[i] / total_count
                    p_j = context_count[j] / total_count
                    pmi = np.log2(p_ij / (p_i * p_j))
                    ppmi_matrix[i][j] = max(pmi, 0)
        return ppmi_matrix

    def generate_embeddings(self):
        embeddings = {}
        for word, idx in self.vocab_index.items():
            embeddings[word] = self.ppmi_matrix[idx]
        return embeddings

    def Apply_SVD(self, k):
        U, S, Vt = np.linalg.svd(self.ppmi_matrix)
        S = np.diag(S)
        S = S[:k, :k]
        U = U[:, :k]
        Vt = Vt[:k, :]
        svd_matrix = np.dot(U, S)
        return svd_matrix

    def get_embedding(self, word):
        if word in self.vocab_index:
            word_idx = self.vocab_index[word]
            return self.svd_embeddings[word_idx]
        else:
            raise ValueError(f"Word '{word}' not found in vocabulary.")

    def save_tokenizer(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.tokenizer, f)
class Glove:
    def __init__(self, doc_path, glove_path, embedding_dim=300, tokenizer_path=None):
        self.docs = self.read_file(doc_path)
        self.embedding_dim = embedding_dim
        self.glove_embeddings = self.load_glove_embeddings(glove_path)
        # self.embedding_matrix = self.create_embedding_matrix()

    def load_glove_embeddings(self, glove_path):
        embeddings = {}
        with open(glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')
                embeddings[word] = vector
        return embeddings

    def read_file(self, doc_path):
        with open(doc_path, 'r') as f:
            docs = []
            for line in f:
                parsed_line = json.loads(line.strip())
                if "train.SRC" in parsed_line:
                    docs.append(parsed_line["train.SRC"])
            return docs

    def create_embedding_matrix(self):
        embedding_matrix = np.zeros((29, self.embedding_dim))
        for word, i in self.word_index.items():
            embedding_vector = self.glove_embeddings.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
            else:
                embedding_matrix[i] = np.random.uniform(-0.01, 0.01, self.embedding_dim)  # For OOV words
        return embedding_matrix

    def save_tokenizer(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.tokenizer, f)

