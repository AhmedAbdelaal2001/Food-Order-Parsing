import tensorflow as tf
import numpy as np
import gensim.downloader as api
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from gensim.models import KeyedVectors
import os
from keras.preprocessing.sequence import pad_sequences
import os
import sys

# Add the parent directory to the Python path
parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_directory)

from Model import Model

class LSTM(Model):
    def __init__(self, model_path, word2vec_path):
        self.model = tf.keras.models.load_model(model_path)
        self.features_extractor = FeaturesExtractor(word2vec_path)

    def predict_labels(self, sentence):
        return (sentence.split(), model_predict(self.model, sentence, self.features_extractor))
    


def model_predict(model, sentence, embedder):
    indices_to_labels = {
        0: "OTHER",
        1: "PIZZA_BEGIN",
        2: "PIZZA_INTERMEDIATE",
        3: "DRINK_BEGIN",
        4: "DRINK_INTERMEDIATE"
    }

    sentence_embeddings = embedder.sentence_to_embeddings(sentence)
    sentence_padded = pad_sequences([sentence_embeddings], dtype='float32', padding='post',value=100)
    predictions = model.predict(sentence_padded, verbose=0)
    predicted_labels = np.argmax(predictions, axis=-1)[0]
    predicted_labels = predicted_labels[:len(sentence.split())]

    # Map each predicted integer label to its corresponding string label
    mapped_labels = [indices_to_labels[label] for label in predicted_labels]
    return mapped_labels


class FeaturesExtractor:
    def __init__(self,model_path):
        self.word2vec_model = self._load_or_download_model(model_path)
        ###################################################################################
        self.styles_list = [
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

        self.styles_avg = self.calculate_average_embedding(self.styles_list)
        ####################################################################################
        self.sizes_list = [
    "small", "medium", "large", "extra large", "regular", "party size",
    "party sized", "party - sized", "party - size", "lunch size",
    "lunch sized", "lunch - sized", "lunch - size", "personal size",
    "personal", "personal sized", "personal - sized", "personal - size"
        ]
        self.sizes_avg = self.calculate_average_embedding(self.sizes_list)


        ####################################################################################
        self.drinks_list = [
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
        self.drinks_avg = self.calculate_average_embedding(self.drinks_list)

        ######################################################################################

        self.quantities_list = [
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
        self.quantities_avg = self.calculate_average_embedding(self.quantities_list)

        #########################################################################################

        self.toppings_list = [
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

        self.toppings_avg = self.calculate_average_embedding(self.toppings_list)

        ############################################################################################################
        self.numbers_list = ["1","a pizza","a drink", "a small","a medium", "a large", "one", "just one", "only one", "two", "2", "three", "3", "four", "4", "five", "5", "six", "6", "seven", "7", "eight", "8", "nine", "9", "ten", "10", "eleven", "11", "twelve", "12", "thirteen", "13", "fourteen", "14", "fifteen", "15"]
        self.numbers_avg = self.calculate_average_embedding(self.numbers_list)


    def calculate_average_embedding(self,topping_list):
        embeddings = []
        for topping in topping_list:
            words = topping.split()  # Split topping into words
            for word in words:
                if word in self.word2vec_model:
                    embeddings.append(self.word2vec_model[word])
        # Calculate the average embedding for all toppings
        if embeddings:
            return np.mean(embeddings, axis=0)
        else:
            return None

    def classify_with_average_embedding(self, candidate, avg):
        similarity = 0
        words = candidate.split()  # Split the candidate into words
        for word in words:
            word_embeddings = [np.zeros(300)]
            if word in self.word2vec_model:
                word_embeddings =  [self.word2vec_model[word]]
        if word_embeddings:
            candidate_embedding = np.mean(word_embeddings, axis=0)
            similarity = cosine_similarity([avg], [candidate_embedding])[0][0]
        return similarity

    def is_style(self,sentence, i):
        window_size =3
        for j in range(-window_size, window_size + 1):  # Check from -3 to +3 words
            if j != 0:  # Skip the current word itself
                new_i = i + j
                if 0 <= new_i < len(sentence):  # Ensure we don't go out of bounds
                    phrase = ' '.join([sentence[k].lower() for k in range(max(0, new_i - window_size), min(len(sentence), new_i + window_size + 1))])
                    if phrase in self.styles_list:
                        return 1
        ret_val = self.classify_with_average_embedding(sentence[i],self.styles_avg)
        return ret_val


    def is_size(self,sentence, i):
        for window_size in range(0, 4):  # Check window sizes 1, 2, and 3
            for j in range(-window_size, window_size + 1):  # Check from -window_size to +window_size words
                # Skip the current word itself
                    new_i = i + j
                    if 0 <= new_i < len(sentence):  # Ensure we don't go out of bounds
                        # Construct the phrase by checking words from new_i - window_size to new_i + window_size
                        phrase = ' '.join([sentence[k].lower() for k in range(max(0, new_i - window_size), min(len(sentence), new_i + window_size + 1))])
                        if phrase in self.sizes_list:  # Check if the phrase is in sizes_list
                            return 1
        ret_val = self.classify_with_average_embedding(sentence[i],self.sizes_avg)
        return ret_val


    def is_drink(self,sentence, i):
        for window_size in range(0, 4):  # Check window sizes 1, 2, and 3
            for j in range(-window_size, window_size + 1):  # Check from -window_size to +window_size words
                # Skip the current word itself
                    new_i = i + j
                    if 0 <= new_i < len(sentence):  # Ensure we don't go out of bounds
                        # Construct the phrase by checking words from new_i - window_size to new_i + window_size
                        phrase = ' '.join([sentence[k].lower() for k in range(max(0, new_i - window_size), min(len(sentence), new_i + window_size + 1))])

                        if phrase in self.drinks_list:  # Check if the phrase is in sizes_list
                            return 1
        ret_val = self.classify_with_average_embedding(sentence[i],self.drinks_avg)
        return ret_val

    def is_quantity(self,sentence, i):
        for window_size in range(0, 4):  # Check window sizes 1, 2, and 3
            for j in range(-window_size, window_size + 1):  # Check from -window_size to +window_size words
                # Skip the current word itself
                    new_i = i + j
                    if 0 <= new_i < len(sentence):  # Ensure we don't go out of bounds
                        # Construct the phrase by checking words from new_i - window_size to new_i + window_size
                        phrase = ' '.join([sentence[k].lower() for k in range(max(0, new_i - window_size), min(len(sentence), new_i + window_size + 1))])
                        if phrase in self.quantities_list:  # Check if the phrase is in sizes_list
                            return 1
        ret_val = self.classify_with_average_embedding(sentence[i],self.quantities_avg)
        return ret_val

    def is_topping(self,sentence, i):
        for window_size in range(0, 4):  # Check window sizes 1, 2, and 3
            for j in range(-window_size, window_size + 1):  # Check from -window_size to +window_size words
                # Skip the current word itself
                    new_i = i + j
                    if 0 <= new_i < len(sentence):  # Ensure we don't go out of bounds
                        # Construct the phrase by checking words from new_i - window_size to new_i + window_size
                        phrase = ' '.join([sentence[k].lower() for k in range(max(0, new_i - window_size), min(len(sentence), new_i + window_size + 1))])

                        if phrase in self.toppings_list:  # Check if the phrase is in sizes_list
                            return 1
        ret_val = self.classify_with_average_embedding(sentence[i],self.toppings_avg)
        return ret_val

    def is_number(self,sentence, i):
        for window_size in range(0, 4):  # Check window sizes 1, 2, and 3
            for j in range(-window_size, window_size + 1):  # Check from -window_size to +window_size words
                # Skip the current word itself
                    new_i = i + j
                    if 0 <= new_i < len(sentence):  # Ensure we don't go out of bounds
                        # Construct the phrase by checking words from new_i - window_size to new_i + window_size
                        phrase = ' '.join([sentence[k].lower() for k in range(max(0, new_i - window_size), min(len(sentence), new_i + window_size + 1))])

                        if phrase in self.numbers_list:  # Check if the phrase is in sizes_list
                            return 1
        ret_val = self.classify_with_average_embedding(sentence[i],self.numbers_avg)
        return ret_val
    #######################################################################################


    def word2features(self,sentence, i):
        sentence[i] = sentence[i].lower()
        features = [0,0,0,0,0,0]
        # Initialize the features for each category
        style = self.is_style(sentence, i)
        if style == 1:
            features[0] = 1
            return np.array(features)

        size = self.is_size(sentence, i)
        if size == 1:
            features[1] = 1
            return np.array(features)


        drink = self.is_drink(sentence, i)
        if drink == 1:
            features[2] = 1
            return np.array(features)


        quantity = self.is_quantity(sentence, i)
        if quantity ==1:
            features[3] = 1
            return np.array(features)

        topping = self.is_topping(sentence, i)
        if topping == 1:
            features[4] = 1
            return np.array(features)

        number = self.is_number(sentence, i)
        if number == 1:
            features[5] =1
            return np.array(features)
        # Store all feature values in a list
        temp_features = [style, size, drink, quantity, topping, number]

        # If no feature is set to 1, choose the max feature (if > 0.51)
        max_value = max(temp_features)
        if max_value > 0.51:
            features[temp_features.index(max_value)] = 1  # Set the feature with max value to 1

        return np.array(features)


    ##########################################################
    def sentence_to_embeddings(self,sentence):
        embeddings = []
        for index, word in enumerate(sentence.split()):
            categories_features = self.word2features(sentence.split(),index)

            if word in self.word2vec_model:
                features = self.word2vec_model[sentence.split()[index]]
            else:
                features = np.zeros(300)
            embeddings.append(np.concatenate((features,np.array(categories_features))))
        return np.array(embeddings)

    #########################################################
    def _load_or_download_model(self,model_path):
        if os.path.exists(model_path):
            print("Loading word2vec model from disk...")
            return KeyedVectors.load(model_path)
        else:
            print("Downloading word2vec model...")
            # Download and save the model
            model = api.load("word2vec-google-news-300")

            return model
