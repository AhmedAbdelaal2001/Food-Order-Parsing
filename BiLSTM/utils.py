import json
import re
import numpy as np
from enum import Enum


class Label(Enum):
    B = 0
    I = 1
    E = 2
    N = 3

def remove_uppercase_words(input_str):
    # Match all uppercase words and replace them with an empty string
    result = re.sub(r'\b(?!PIZZAORDER\b|DRINKORDER\b)[A-Z_]+\b', '', input_str)
    
    # Remove extra spaces caused by the replacements
    result = re.sub(r'\s+', ' ', result).strip()
    
    return result.split()

def get_important_words(input: str, output: str) -> dict[int, str]:   
    input = input.split()
    output = remove_uppercase_words(output)

    i = 0
    j = 0

    words_with_brackets = {}

    while i < len(input):
        while output[j] in ["(", ")", "(PIZZAORDER", "(DRINKORDER"]:
            j += 1

        if output[j-1] == "(":
            temp_j = j
            temp_i = i
            temp_dict = {}
            while output[temp_j] not in ["(", ")", "(PIZZAORDER", "(DRINKORDER"]:
                temp_dict[temp_i] = (output[temp_j], temp_j)
                temp_j += 1
                temp_i += 1

            if output[temp_j] == ")":
                words_with_brackets.update(temp_dict)
            
            i = temp_i
            j = temp_j
        
        else:
            i+=1
            j+=1

    return words_with_brackets

def create_label_from_sample(input: str, output: str) -> str:
    custom_output = remove_uppercase_words(output)
    important_words = get_important_words(input, output)
    
    label = []

    for i in range(len(input.split())):
        if i in important_words:
            j = important_words[i][1]
            
            if j > 1 and custom_output[j-2] in ["(PIZZAORDER", "(DRINKORDER"]:
                label.append(Label.B.value)
            else: label.append(Label.I.value)
            
        else: label.append(Label.N.value)

    return label

def read_dataset_from_json(file_path: str) -> list[dict[str, str]]:
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]
    
def load_data():
    train_data = read_dataset_from_json("../dataset/preprocessed_PIZZA_train.json")
    dev_data = read_dataset_from_json("../dataset/preprocessed_PIZZA_dev.json")
    test_data = read_dataset_from_json("../dataset/preprocessed_PIZZA_test.json")

    X_train, y_train = [sample["train.SRC"] for sample in train_data], [sample["train.TOP"] for sample in train_data]
    X_dev, y_dev = [sample["dev.SRC"] for sample in dev_data], [sample["dev.TOP"] for sample in dev_data]
    X_test, y_test = [sample["test.SRC"] for sample in test_data], [sample["test.TOP"] for sample in test_data]

    return np.array(X_train), np.array(y_train), np.array(X_dev), np.array(y_dev), np.array(X_test), np.array(y_test)

def get_word2vec(word, word2vec):
    if word in word2vec:
        return word2vec[word]
    else:
        return np.zeros_like(word2vec["pizza"])
    
def sentence_to_word2vec(sentence, word2vec):
    return np.array([get_word2vec(word, word2vec) for word in sentence.split()])

def create_lstm_labels(X, y):
    y_lstm = [create_label_from_sample(X[i], y[i]) for i in range(len(X))]
    
    return y_lstm

def write_lstm_dataset_to_json(file_path: str, X, y):
    with open(file_path, 'w') as file:
        for i in range(len(X)):
            json.dump({"X": X[i], "y": y[i]}, file)
            file.write('\n')

def load_lstm_dataset_from_json(file_path: str, size=None):
    X, y = [], []
    with open(file_path, 'r') as file:
        for i, line in enumerate(file):
            if size is not None and i >= size:
                break
            sample = json.loads(line)
            X.append(sample["X"])
            y.append(sample["y"])

    return X, y