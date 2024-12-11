import numpy as np
from nltk.tokenize import word_tokenize
from transformers import BertTokenizer, BertModel
import torch
class PPMI :
    def __init__(self, doc_path, window_size=2,k=100):
        self.docs = self.read_file(doc_path)
        self.tokenized_docs = self.tokenize(self.docs)
        self.window_size = window_size
        self.cooccurrence_matrix, self.vocab_index = self.build_cooccurrence_matrix(self.tokenized_docs, window_size)
        self.ppmi_matrix = self.build_ppmi_matrix(self.cooccurrence_matrix)
        self.embeddings = self.generate_embeddings()
        self.embedding_dim = self.ppmi_matrix.shape[1]
        self.svd_embeddings = self.Apply_SVD(k)
        
    def read_file(self, doc_path):
        try:
            with open(doc_path, 'r') as f:
                docs = [line.strip().strip('[]').strip('"') for line in f]
            if not docs:
                raise ValueError("Document file is empty.")
            return docs
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {doc_path}")

    def tokenize(self, docs):
        return [word_tokenize(doc) for doc in docs]
        
    def build_cooccurrence_matrix(self,sentences, window_size=2): # window_size is the number of words to the left and right of the target word
        vocab = set()
        for sentence in sentences:
            vocab.update(sentence)
        vocab = list(vocab)
        vocab_index = {word: i for i, word in enumerate(vocab)} # create a dictionary with word as key and index as value
        cooccurrence_matrix = np.zeros((len(vocab), len(vocab)))
        
        for sentence in sentences:
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
        context_count = np.sum(cooccurrence_matrix, axis=0) # sum of each column
        ppmi_matrix = np.zeros_like(cooccurrence_matrix) # create a zero matrix with the same shape as cooccurrence_matrix
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
        svd_matrix =np.dot(U, S)
        return svd_matrix
    def get_embedding(self, word):
        if word in self.vocab_index:
            word_idx = self.vocab_index[word]
            return self.svd_embeddings[word_idx]
        else:
            raise ValueError(f"Word '{word}' not found in vocabulary.")
    
class Glove:
    def __init__(self,doc_path, glove_path, embedding_dim=100):
        self.docs = self.read_file(doc_path)
        self.tokenized_docs = self.tokenize(self.docs)
        self.glove_embeddings = self.load_glove_embeddings(glove_path)
        self.embedding_dim =embedding_dim
        self.embeddings = self.process_embeddings()
    def load_glove_embeddings(self,glove_path):
        embeddings = {}
        with open(glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')
                embeddings[word] = vector
        return embeddings      
    def read_file(self, doc_path):
        try:
            with open(doc_path, 'r') as f:
                docs = [line.strip().strip('[]').strip('"') for line in f]
            if not docs:
                raise ValueError("Document file is empty.")
            return docs
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {doc_path}")

    def tokenize(self, docs):
        return [word_tokenize(doc) for doc in docs]
    
    def process_embeddings(self):
        embeddings = {}
        for doc in self.tokenized_docs:
            for word in doc:
                if word in self.glove_embeddings:
                    embeddings[word] = self.glove_embeddings[word]
                else:
                    embeddings[word]=np.zeros(self.embedding_dim)
        return embeddings
class BERT:
    def __init__(self, doc_path, model_name='bert-base-uncased'):
        self.docs = self.read_file(doc_path)
        self.tokenizer, self.model = self.load_bert_embeddings(model_name)
        self.embedding_dim = 768  # Fixed dimension for BERT base model
        self.embeddings = self.process_embeddings()
    
    def load_bert_embeddings(self, model_name):
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name)
        model.eval()  # Set the model to evaluation mode so that the embeddings can not be updated
        return tokenizer, model
    
    def read_file(self, doc_path):
        try:
            with open(doc_path, 'r') as f:
                docs = [line.strip() for line in f]
            if not docs:
                raise ValueError("Document file is empty.")
            return docs
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {doc_path}")
    
    def process_embeddings(self):
        embeddings = {}
        for doc in self.docs:
            # Tokenize the document and get input tensors
            inputs = self.tokenizer(doc, return_tensors='pt', padding=False, truncation=True)
            
            with torch.no_grad():  # Disable gradient calculation
                outputs = self.model(**inputs)
            
            last_hidden_states = outputs.last_hidden_state  # Shape: (1, num_tokens, embedding_dim)

            # Convert input IDs to tokens
            tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'].squeeze(0))

            # Store embeddings for each token
            doc_embeddings = {}
            for i, token in enumerate(tokens):
                doc_embeddings[token] = last_hidden_states[0, i, :].detach().numpy()
            
            embeddings[doc] = doc_embeddings
        return embeddings
    def get_embedding(self, doc, token):
        if doc in self.embeddings:
            doc_embeddings = self.embeddings[doc]
            if token in doc_embeddings:
                return doc_embeddings[token]
            else:
                raise ValueError(f"Token '{token}' not found in document '{doc}'.")
        else:
            raise ValueError(f"Document '{doc}' not found in embeddings.")


    