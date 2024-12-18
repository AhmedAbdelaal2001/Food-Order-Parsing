import re

class Node:
    """
    Represents a node in the labeled hierarchical structure.
    Each node has a label and can have child nodes or text.
    """
    def __init__(self, label):
        self.label = label
        self.children = []  # List of child Node instances
        self.text = None    # Text content if the node is a leaf

    def to_string(self):
        """
        Recursively converts the node and its children back into the bracketed string format.
        """
        if self.text and not self.children:
            # Leaf node with text
            return f"({self.label} {self.text} )"
        elif self.children:
            # Node with child labels
            children_str = ' '.join(child.to_string() for child in self.children)
            return f"({self.label} {children_str} )"
        else:
            # Node without children or text
            return f"({self.label} )"

def tokenize(top_string):
    """
    Splits the TOP string into tokens: '(', ')', and words.
    """
    return re.findall(r'\(|\)|[^\s()]+', top_string)

def parse(tokens, index):
    """
    Recursively parses tokens to build the hierarchical structure.
    
    Parameters:
    - tokens: List of tokens from the TOP string.
    - index: Current position in the token list.
    
    Returns:
    - node: The parsed Node object.
    - index: Updated position after parsing.
    """
    if tokens[index] != '(':
        # Not a label, skip
        return None, index

    label = tokens[index + 1]
    node = Node(label)
    index += 2  # Move past '(' and label

    texts = []        # Collect text tokens
    has_children = False  # Flag to indicate presence of child labels

    while index < len(tokens):
        token = tokens[index]
        if token == '(':
            # Found a child label; parse it recursively
            child, index = parse(tokens, index)
            if child:
                node.children.append(child)
                has_children = True
        elif token == ')':
            index += 1  # Move past ')'
            if not has_children and texts:
                # If no child labels, set the text content
                node.text = ' '.join(texts)
            return node, index
        else:
            # Text token
            if not has_children:
                # Only collect text if no child labels have been found
                texts.append(token)
            index += 1

    return node, index

def generate_top_decoupled_from_top(top_string):
    """
    Converts a TOP string into a TOP-DECOUPLED string by removing redundant tokens.
    
    Parameters:
    - top_string (str): The input TOP string.
    
    Returns:
    - decoupled_str (str): The resulting TOP-DECOUPLED string.
    """
    tokens = tokenize(top_string)
    node, _ = parse(tokens, 0)
    if node:
        return node.to_string()
    else:
        return ""


def segment_orders(words, labels):
    """
    Segments the input sentence into pizza and drink orders based on NER labels.
    
    Args:
        sentence (str): The input sentence to process.
        ner_pipeline_instance (transformers.pipeline): The NER pipeline.
    
    Returns:
        tuple: Two lists containing pizza orders and drink orders respectively.
    """
    
    orders = []
    
    current_pizza = []
    current_drink = []
    
    in_pizza = False
    in_drink = False
    
    for word, label in zip(words, labels):
        # Handle Pizza Orders
        if label == 'PIZZA_BEGIN':
            if in_pizza and current_pizza:
                orders.append((' '.join(current_pizza), True))
                current_pizza = []
            in_pizza = True
            current_pizza.append(word)
        elif label == 'PIZZA_INTERMEDIATE' and in_pizza:
            current_pizza.append(word)
        else:
            if in_pizza and current_pizza:
                orders.append((' '.join(current_pizza), True))
                current_pizza = []
            in_pizza = False
        
        # Handle Drink Orders
        if label == 'DRINK_BEGIN':
            if in_drink and current_drink:
                orders.append((' '.join(current_drink), False))
                current_drink = []
            in_drink = True
            current_drink.append(word)
        elif label == 'DRINK_INTERMEDIATE' and in_drink:
            current_drink.append(word)
        else:
            if in_drink and current_drink:
                orders.append((' '.join(current_drink), False))
                current_drink = []
            in_drink = False
    
    # Append any remaining orders after the loop
    if in_pizza and current_pizza:
        orders.append((' '.join(current_pizza), True))
    if in_drink and current_drink:
        orders.append((' '.join(current_drink), False))
    
    return orders


def generate_top_decoupled(sentence, labels, is_pizza_order):
    output_sentence = ""
    tokens = sentence.split()
    i = 0
    while i < len(tokens):
        if labels[i] == "OTHER": 
            i += 1
            continue
        if '-' in labels[i]:
            index = labels[i].find('-')
            parent_identifier = labels[i][:index]
            sub_tokens = []
            sub_labels = []
            while i < len(tokens) and labels[i][:index] == parent_identifier:
                sub_tokens.append(tokens[i])
                sub_labels.append(labels[i][index+1:])
                i += 1
                continue
            nested_part_string = generate_top_decoupled(' '.join(sub_tokens), sub_labels, -1)
            output_sentence += ('(' + parent_identifier + ' ' + nested_part_string + ') ')
            continue

        curr_label = labels[i]
        curr_element = []
        j = 0
        while i+j < len(labels) and labels[i+j] == curr_label:
            curr_element.append(tokens[i+j])
            j += 1
            if curr_label == "TOPPING_BEGIN": curr_label = "TOPPING_INTERMEDIATE"
            elif curr_label == "STYLE_BEGIN": curr_label = "STYLE_INTERMEDIATE"
        i = i + j - 1
        j = 0
        curr_element_string = ' '.join(curr_element)
        if curr_label == "TOPPING_INTERMEDIATE": curr_label = "TOPPING"
        elif curr_label == "STYLE_INTERMEDIATE": curr_label = "STYLE"
        output_sentence += ('(' + curr_label + ' ' + curr_element_string + ' ) ')

        i += 1

    if is_pizza_order == -1:
        return output_sentence
    
    if is_pizza_order == 0: 
        identifier = '(PIZZAORDER '
    else:
        identifier = '(DRINKORDER '

    output_sentence = identifier + output_sentence + ')'
    return output_sentence