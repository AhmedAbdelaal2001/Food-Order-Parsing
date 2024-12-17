import json
import re
import numpy as np
from enum import Enum
from sklearn.metrics.pairwise import cosine_similarity



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

def classify_with_average_embedding(candidate,avg, wv):
    similarity = 0
    words = candidate.split()  # Split the candidate into words
    word_embeddings = [wv[word] for word in words if word in wv]
    if word_embeddings:
        candidate_embedding = np.mean(word_embeddings, axis=0)
        similarity = cosine_similarity([avg], [candidate_embedding])[0][0]
    return similarity

def calculate_average_embedding(topping_list, wv):
    embeddings = []
    for topping in topping_list:
        words = topping.split()  # Split topping into words
        for word in words:
           if word in wv:
              embeddings.append(wv[word])

    # Calculate the average embedding for all toppings
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return None
    
styles_list = [
    "cauliflower crust", "cauliflower crusts", "gluten free crust", "gluten-free crust",
    "gluten free crusts", "gluten-free crusts", "keto crust", "keto crusts",
    "sourdough crust", "sourdough crusts", "stuffed crust", "stuffed crusts",
    "thick crust", "thick crusts", "high rise dough", "thin crust", "thin crusts",
    "vegan", "vegetarian", "veggie", "supreme", "new york style", "big new yorker",
    "napolitana", "napolitan", "neapolitan", "mediterranean", "med", "mexican",
    "big meat", "meat lover", "meat lovers", "meatlover", "meatlovers", "every meat",
    "all meat", "margherita", "margarita", "hawaiian", "deep dish", "deepdish",
    "pan", "combination", "chicago style", "chicago", "all the cheese", "all cheese",
    "cheese lover", "cheese lovers", "all the toppings", "everything", "with the works",
    "every topping", "all the vegetables", "all veggies"
]


def is_style(sentence, i, wv):
    styles_avg = calculate_average_embedding(styles_list, wv)
    for j in range(1, 5):  # Check up to 3 words ahead
        if i + j <= len(sentence):  # Ensure we don't go out of bounds
            phrase = ' '.join([sentence[k].lower() for k in range(i, i + j)])
            if phrase in styles_list:
                return 1

    # Check words behind (up to 3 words)
    for j in range(1, 5):  # Check up to 3 words behind
        if i - j >= 0:  # Ensure we don't go out of bounds
            phrase = ' '.join([sentence[k].lower() for k in range(i - j, i + 1)])
            if phrase in styles_list:
                return 1
    ret_val = classify_with_average_embedding(sentence[i],styles_avg, wv)
    return ret_val if ret_val > 0.7 else 0

sizes_list = [
    "small", "medium", "large", "extra large", "regular", "party size",
    "party sized", "party - sized", "party - size", "lunch size",
    "lunch sized", "lunch - sized", "lunch - size", "personal size",
    "personal", "personal sized", "personal - sized", "personal - size"
]

def is_size(sentence, i, wv):
    sizes_avg = calculate_average_embedding(sizes_list, wv)
    for j in range(1, 5):  # Check up to 3 words ahead
        if i + j <= len(sentence):  # Ensure we don't go out of bounds
            phrase = ' '.join([sentence[k].lower() for k in range(i, i + j)])
            if phrase in sizes_list:
                return 1

    # Check words behind (up to 3 words)
    for j in range(1, 5):  # Check up to 3 words behind
        if i - j >= 0:  # Ensure we don't go out of bounds
            phrase = ' '.join([sentence[k].lower() for k in range(i - j, i + 1)])
            if phrase in sizes_list:
                return 1
    ret_val = classify_with_average_embedding(sentence[i],sizes_avg,wv)
    return ret_val if ret_val > 0.7 else 0

drinks_list = [
    "7 up", "7 ups", "seven up", "seven ups", "cherry coke", "cherry cokes",
    "cherry pepsi", "cherry pepsis", "coffee", "coffees", "coke", "cokes",
    "coke zero", "coke zeros", "coke zeroes", "dr pepper", "dr peppers",
    "dr peper", "dr pepers", "doctor peppers", "doctor pepper", "doctor pepers",
    "doctor peper", "fanta", "fantas", "ginger ale", "ginger ales",
    "ice tea", "iced tea", "ice teas", "iced teas", "lemon ice tea",
    "lemon iced tea", "lemon ice teas", "lemon iced teas", "mountain dew",
    "mountain dews", "pellegrino", "pellegrinos", "san pellegrino",
    "san pellegrinos", "pepsi", "pepsis", "perrier", "perriers",
    "pineapple soda", "pineapple sodas", "sprite", "sprites", "water",
    "waters", "diet pepsi", "diet pepsis", "diet coke", "diet cokes",
    "diet sprite", "diet sprites", "diet ice tea", "diet iced tea",
    "diet ice teas", "diet iced teas"
]

def is_drink(sentence, i, wv):
    drinks_avg = calculate_average_embedding(drinks_list, wv)
    for j in range(1, 5):  # Check up to 3 words ahead
        if i + j <= len(sentence):  # Ensure we don't go out of bounds
            phrase = ' '.join([sentence[k].lower() for k in range(i, i + j)])
            if phrase in drinks_list:
                return 1

    # Check words behind (up to 3 words)
    for j in range(1, 5):  # Check up to 3 words behind
        if i - j >= 0:  # Ensure we don't go out of bounds
            phrase = ' '.join([sentence[k].lower() for k in range(i - j, i + 1)])
            if phrase in drinks_list:
                return 1
    ret_val = classify_with_average_embedding(sentence[i],drinks_avg,wv)
    return ret_val if ret_val > 0.7 else 0

quantities_list = [
    "light", "go light on the", "go light on", "light on the", "light on",
    "little", "a little", "just a little", "just a bit", "only a little",
    "only a bit", "not a lot of", "not a lot", "not much", "not many",
    "a little bit", "a little bit of", "a drizzle of", "a drizzle",
    "just a drizzle", "just a drizzle of", "only a drizzle",
    "only a drizzle of", "no more than a drizzle", "no more than a drizzle of",
    "just a tiny bit of", "a tiny bit of", "go heavy on", "go heavy on the",
    "heavy on", "heavy on the", "lots of", "a lot of", "a whole lot of",
    "a bunch of", "a whole bunch of", "extra", "lot of","lot"
]

def is_quantity(sentence, i, wv):
    quantities_avg = calculate_average_embedding(quantities_list, wv)
    for j in range(1, 5):  # Check up to 3 words ahead
        if i + j <= len(sentence):  # Ensure we don't go out of bounds
            phrase = ' '.join([sentence[k].lower() for k in range(i, i + j)])
            if phrase in quantities_list:
                return 1

    # Check words behind (up to 3 words)
    for j in range(1, 5):  # Check up to 3 words behind
        if i - j >= 0:  # Ensure we don't go out of bounds
            phrase = ' '.join([sentence[k].lower() for k in range(i - j, i + 1)])
            if phrase in quantities_list:
                return 1
    ret_val = classify_with_average_embedding(sentence[i],quantities_avg,wv)
    return ret_val if ret_val > 0.5 else 0

toppings_list = [
    "alfredo chicken", "american cheese", "anchovy", "anchovies", "artichoke", "artichokes",
    "arugula", "bacon", "bacons", "apple wood bacon", "applewood bacon", "balsamic glaze",
    "balzamic glaze", "banana pepper", "banana peppers", "basil", "bay leaves", "bbq chicken",
    "barbecue chicken", "bbq pulled pork", "barbecue pulled pork", "bbq sauce", "barbecue sauce",
    "bean", "beans", "beef", "ground beef", "broccoli", "brocoli", "buffalo chicken", "buffalo mozzarella",
    "buffalo mozarella", "buffalo sauce", "caramelized onions", "caramelized red onions", "caramelized onion",
    "caramelized red onion", "carrot", "carrots", "cheddar cheese", "cheese", "cheeseburger", "cherry tomato",
    "cherry tomatoes", "chicken", "chickens", "chorizo", "chorrizo", "cumin", "dried pepper", "dried peppers",
    "dried tomato", "dried tomatoes", "feta cheese", "feta", "fried onion", "fried onions", "garlic",
    "garlic powder", "green olive", "green olives", "green pepper", "green peppers", "grilled chicken",
    "grilled pineapple", "ham", "hams", "hot pepper", "hot peppers", "italian sausage", "jalapeno pepper",
    "jalapeno", "jalapeno peppers", "jalapenos", "kalamata olive", "kalamata olives", "lettuce",
    "low fat cheese", "meatball", "meatballs", "mozzarella cheese", "mozarella cheese", "mozzarella",
    "mozarella", "mushroom", "mushrooms", "olive oil", "olives", "olive", "black olive", "black olives",
    "onions", "onion", "oregano", "parmesan cheese", "parmesan", "parsley", "pea", "peas", "pecorino cheese",
    "pecorino", "pepperoni", "peppperoni", "pepperonis", "peppperonis", "peperoni", "peperonis",
    "peperroni", "peperonni", "peperronni", "peppers", "pepper", "pesto", "pestos", "pesto sauce",
    "pickle", "pickles", "pineapple", "pineapples", "pineaple", "pineaples", "ranch sauce", "red onion",
    "red onions", "red pepper flake", "red pepper flakes", "red peppers", "red pepper", "ricotta cheese",
    "ricotta", "roasted chicken", "roasted garlic", "roasted pepper", "roasted peppers", "roasted red pepper",
    "roasted red peppers", "roasted green pepper", "roasted green peppers", "roasted tomato",
    "roasted tomatoes", "rosemary", "salami", "sauce", "sausage", "sausages", "shrimp", "shrimps",
    "spiced sausage", "spicy red sauce", "spinach", "tomato sauce", "tomato", "tomatoes", "tuna", "tunas",
    "vegan pepperoni", "white onion", "white onions", "yellow pepper", "yellow peppers","green","red","oil"
]


def is_topping(sentence, i, wv):
    toppings_avg = calculate_average_embedding(toppings_list, wv)
    if sentence[i] == "pizza" or sentence[i] == "pizzas":
      return 0
    for j in range(1, 5):  # Check up to 3 words ahead
        if i + j <= len(sentence):  # Ensure we don't go out of bounds
            phrase = ' '.join([sentence[k].lower() for k in range(i, i + j)])
            if phrase in toppings_list:
                return 1

    # Check words behind (up to 3 words)
    for j in range(1, 5):  # Check up to 3 words behind
        if i - j >= 0:  # Ensure we don't go out of bounds
            phrase = ' '.join([sentence[k].lower() for k in range(i - j, i + 1)])
            if phrase in toppings_list:
                return 1
    ret_val = classify_with_average_embedding(sentence[i],toppings_avg,wv)
    return ret_val if ret_val > 0.45 else 0


def word2features(sentence, i, wv):
    sentence[i] = sentence[i].lower()
    #style
    style = is_style(sentence,i,wv)

    #size
    size = 0
    if(style != 1):
      size = is_size(sentence,i,wv)
      if(size > style):
        style = 0


    #drink
    drink = is_drink(sentence,i,wv)

    #quantity
    quantity = is_quantity(sentence,i,wv)


    #topping
    topping = 0
    if(drink != 1):
      topping = is_topping(sentence,i,wv)
      if(topping > drink):
        drink = 0
      if(style ==1 ):
        topping = 0



    features = {
        'word': sentence[i],


        # Topping feature
        'is_topping': topping,

        # Quantifier feature
        'quantifier': quantity,

        # Style feature
        'style': style,

        # Size feature
        'size': size,
        #Drink
        'drink': drink
    }
    return np.array([topping, quantity, style, size,drink])


