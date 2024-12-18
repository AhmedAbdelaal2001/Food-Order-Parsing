from Model import *
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import gensim.downloader as api
from gensim.models import KeyedVectors
from tensorflow.keras.preprocessing.sequence import pad_sequences
class BiLSTM_Module2(Model):
    word2vec=None
    def __init__(self, model_path):
        self.model = load_model(model_path)
        if BiLSTM_Module2.word2vec is None:
            BiLSTM_Module2.word2vec = api.load("word2vec-google-news-300")
        self.labels_to_indices ={"COMPLEX_TOPPING-QUANTITY": 0,
        "COMPLEX_TOPPING-TOPPING_BEGIN":1,
        'COMPLEX_TOPPING-TOPPING_INTERMEDIATE':2,
        'CONTAINERTYPE':3,
        'DRINKTYPE':4,
        'NOT-COMPLEX_TOPPING-QUANTITY':5,
        'NOT-COMPLEX_TOPPING-TOPPING_BEGIN':6,
        'NOT-STYLE':7,
        'NOT-TOPPING_BEGIN':8,
        'NOT-TOPPING_INTERMEDIATE':9,
        'NUMBER':10,
        'OTHER':11,
        'SIZE':12,
        'STYLE':13,
        'TOPPING_BEGIN':14,
        'TOPPING_INTERMEDIATE':15,
        'VOLUME':16}
    def predict_labels(self, sentence):
        sentence_words=sentence.split()
        sentence_length=len(sentence_words)
        embeddings = load_embeddings(sentence)
        padded = pad_sequences([embeddings], maxlen=sentence_length, dtype='float32', padding='post', value=100)
        predictions = self.model.predict(padded)
        predicted_labels = np.argmax(predictions, axis=-1)
        # Map indices back to labels
        index_to_label = {v: k for k, v in self.labels_to_indices.items()}
        predicted_labels_mapped = [
            [index_to_label.get(idx) for idx in seq] for seq in predicted_labels
        ]
        return sentence_words, predicted_labels_mapped
    
def get_word2vec():
    if BiLSTM_Module2.word2vec is None:
        BiLSTM_Module2.word2vec = api.load("word2vec-google-news-300")
    return BiLSTM_Module2.word2vec  
def load_embeddings(sentence):
  word2vec=get_word2vec()
  def classify_with_average_embedding(candidate,avg):
    similarity = 0
    words = candidate.split()  # Split the candidate into words
    for word in words:
        word_embeddings = [np.zeros(300)]
        if word in word2vec:
          word_embeddings =  [word2vec[word]]
    if word_embeddings:
        candidate_embedding = np.mean(word_embeddings, axis=0)
        similarity = cosine_similarity([avg], [candidate_embedding])[0][0]
    return similarity
  def calculate_average_embedding(topping_list):
      embeddings = []
      for topping in topping_list:
          words = topping.split()  # Split topping into words
          for word in words:
              if word in word2vec:
                embeddings.append(word2vec[word])

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

  styles_avg = calculate_average_embedding(styles_list)

  def is_style(sentence, i):
      window_size =3
      for j in range(-window_size, window_size + 1):  # Check from -3 to +3 words
          if j != 0:  # Skip the current word itself
              new_i = i + j
              if 0 <= new_i < len(sentence):  # Ensure we don't go out of bounds
                  phrase = ' '.join([sentence[k].lower() for k in range(max(0, new_i - window_size), min(len(sentence), new_i + window_size + 1))])
                  if phrase in styles_list:
                      return 1
      ret_val = classify_with_average_embedding(sentence[i],styles_avg)
      return ret_val


  sizes_list = [
      "small", "medium", "large", "extra large", "regular", "party size",
      "party sized", "party - sized", "party - size", "lunch size",
      "lunch sized", "lunch - sized", "lunch - size", "personal size",
      "personal", "personal sized", "personal - sized", "personal - size"
  ]
  sizes_avg = calculate_average_embedding(sizes_list)

  def is_size(sentence, i):
      for window_size in range(0, 4):  # Check window sizes 1, 2, and 3
          for j in range(-window_size, window_size + 1):  # Check from -window_size to +window_size words
                # Skip the current word itself
                  new_i = i + j
                  if 0 <= new_i < len(sentence):  # Ensure we don't go out of bounds
                      # Construct the phrase by checking words from new_i - window_size to new_i + window_size
                      phrase = ' '.join([sentence[k].lower() for k in range(max(0, new_i - window_size), min(len(sentence), new_i + window_size + 1))])

                      if phrase in sizes_list:  # Check if the phrase is in sizes_list
                          return 1
      ret_val = classify_with_average_embedding(sentence[i],sizes_avg)
      return ret_val

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
  drinks_avg = calculate_average_embedding(drinks_list)

  def is_drink(sentence, i):
      for window_size in range(0, 4):  # Check window sizes 1, 2, and 3
          for j in range(-window_size, window_size + 1):  # Check from -window_size to +window_size words
                # Skip the current word itself
                  new_i = i + j
                  if 0 <= new_i < len(sentence):  # Ensure we don't go out of bounds
                      # Construct the phrase by checking words from new_i - window_size to new_i + window_size
                      phrase = ' '.join([sentence[k].lower() for k in range(max(0, new_i - window_size), min(len(sentence), new_i + window_size + 1))])

                      if phrase in drinks_list:  # Check if the phrase is in sizes_list
                          return 1
      ret_val = classify_with_average_embedding(sentence[i],drinks_avg)
      return ret_val


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
  quantities_avg = calculate_average_embedding(quantities_list)

  def is_quantity(sentence, i):
      for window_size in range(0, 4):  # Check window sizes 1, 2, and 3
          for j in range(-window_size, window_size + 1):  # Check from -window_size to +window_size words
                # Skip the current word itself
                  new_i = i + j
                  if 0 <= new_i < len(sentence):  # Ensure we don't go out of bounds
                      # Construct the phrase by checking words from new_i - window_size to new_i + window_size
                      phrase = ' '.join([sentence[k].lower() for k in range(max(0, new_i - window_size), min(len(sentence), new_i + window_size + 1))])
                      if phrase in quantities_list:  # Check if the phrase is in sizes_list
                          return 1
      ret_val = classify_with_average_embedding(sentence[i],quantities_avg)
      return ret_val
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

  toppings_avg = calculate_average_embedding(toppings_list)

  def is_topping(sentence, i):

      for window_size in range(0, 4):  # Check window sizes 1, 2, and 3
          for j in range(-window_size, window_size + 1):  # Check from -window_size to +window_size words
                # Skip the current word itself
                  new_i = i + j
                  if 0 <= new_i < len(sentence):  # Ensure we don't go out of bounds
                      # Construct the phrase by checking words from new_i - window_size to new_i + window_size
                      phrase = ' '.join([sentence[k].lower() for k in range(max(0, new_i - window_size), min(len(sentence), new_i + window_size + 1))])

                      if phrase in toppings_list:  # Check if the phrase is in sizes_list
                          return 1
      ret_val = classify_with_average_embedding(sentence[i],toppings_avg)
      return ret_val

  numbers_list = ["1","a pizza","a drink", "a small","a medium", "a large", "one", "just one", "only one", "two", "2", "three", "3", "four", "4", "five", "5", "six", "6", "seven", "7", "eight", "8", "nine", "9", "ten", "10", "eleven", "11", "twelve", "12", "thirteen", "13", "fourteen", "14", "fifteen", "15"]
  numbers_avg = calculate_average_embedding(numbers_list)

  def is_number(sentence, i):

      for window_size in range(0, 4):  # Check window sizes 1, 2, and 3
          for j in range(-window_size, window_size + 1):  # Check from -window_size to +window_size words
                # Skip the current word itself
                  new_i = i + j
                  if 0 <= new_i < len(sentence):  # Ensure we don't go out of bounds
                      # Construct the phrase by checking words from new_i - window_size to new_i + window_size
                      phrase = ' '.join([sentence[k].lower() for k in range(max(0, new_i - window_size), min(len(sentence), new_i + window_size + 1))])

                      if phrase in numbers_list:  # Check if the phrase is in sizes_list
                          return 1
      ret_val = classify_with_average_embedding(sentence[i],numbers_avg)
      return ret_val
  def word2features(sentence, i):
      sentence[i] = sentence[i].lower()
      features = [0,0,0,0,0,0]
      # Initialize the features for each category
      style = is_style(sentence, i)
      if style == 1:
        features[0] = 1
        return np.array(features)
      size = is_size(sentence, i)
      if size == 1:
        features[1] = 1
        return np.array(features)
      drink = is_drink(sentence, i)
      if drink == 1:
        features[2] = 1
        return np.array(features)
      quantity = is_quantity(sentence, i)
      if quantity ==1:
        features[3] = 1
        return np.array(features)
      topping = is_topping(sentence, i)
      if topping == 1:
        features[4] = 1
        return np.array(features)
      number = is_number(sentence, i)
      if number == 1:
        features[5] =1
        return np.array(features)
      # Store all feature values in a list
      temp_features = [style, size, drink, quantity, topping, number]
      # If no feature is set to 1, choose the max feature (if > 0.5)
      max_value = max(temp_features)
      if max_value > 0.51:
          features[temp_features.index(max_value)] = 1  # Set the feature with max value to 1
      return np.array(features)


  def sentence_to_embeddings(sentence):
    embeddings = []
    for index, word in enumerate(sentence.split()):
        categories_features = word2features(sentence.split(),index)
        if word in word2vec:
          features = word2vec[sentence.split()[index]]
        else:
          features = np.zeros(300)
        embeddings.append(np.concatenate((features,np.array(categories_features))))
    return np.array(embeddings)

  return sentence_to_embeddings(sentence)
