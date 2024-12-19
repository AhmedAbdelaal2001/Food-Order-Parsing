import re
import json

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

def tokenize(s):
    # Simple tokenizer that splits on spaces and parentheses.
    tokens = []
    current = []
    for char in s:
        if char.isspace():
            if current:
                tokens.append("".join(current))
                current = []
        elif char == '(' or char == ')':
            if current:
                tokens.append("".join(current))
                current = []
            tokens.append(char)
        else:
            current.append(char)
    if current:
        tokens.append("".join(current))
    return tokens

def parse_tokens(tokens):
    # Parse the tokens into a nested list structure.
    # Each time we see '(', we start a new list until we see ')'.
    # Example: (PIZZAORDER (NUMBER two)) -> ['PIZZAORDER', ['NUMBER', 'two']]
    def helper(it):
        lst = []
        for token in it:
            if token == '(':
                lst.append(helper(it))
            elif token == ')':
                return lst
            else:
                lst.append(token)
        return lst

    return helper(iter(tokens))

def singularize(s):
    # Simple heuristic to singularize containers if they end with 's'
    if s.endswith('s'):
        return s[:-1]
    return s

def process_structure(parsed):
    # parsed is a list of top-level elements like ['PIZZAORDER', [...]] repeated.
    # We want to collect PIZZAORDER and DRINKORDER segments.
    pizza_orders = []
    drink_orders = []

    # The top-level structure might look like:
    # [
    #   ['PIZZAORDER', ['NUMBER','two'], ['SIZE','large'], ... ],
    #   ['PIZZAORDER', ['NUMBER','three'], ['SIZE','small'], ... ],
    #   ['DRINKORDER', ['NUMBER','five'], ...],
    #   ...
    # ]

    for element in parsed:
        # element[0] is the type: PIZZAORDER or DRINKORDER, the rest are attributes
        order_type = element[0]
        if order_type == 'PIZZAORDER':
            pizza_orders.append(process_pizza_order(element[1:]))
        elif order_type == 'DRINKORDER':
            drink_orders.append(process_drink_order(element[1:]))

    return {
        "ORDER": {
            "PIZZAORDER": pizza_orders,
            "DRINKORDER": drink_orders
        }
    }

def process_pizza_order(attributes):
    number = None
    size = None
    style = []
    all_topping = []

    for attr in attributes:
        if not attr:
            continue
        key = attr[0]

        if key == 'NUMBER':
            number = attr[1]
        elif key == 'SIZE':
            # Join all following tokens in case SIZE has multiple words
            size = " ".join(attr[1:])
        elif key == 'STYLE':
            style_text = " ".join(attr[1:])
            style.append({"NOT": False, "TYPE": style_text})
        elif key == 'COMPLEX_TOPPING':
            quantity, topping = None, None
            not_flag = False
            for comp in attr[1:]:
                if comp[0] == 'QUANTITY':
                    quantity = " ".join(comp[1:])
                elif comp[0] == 'TOPPING':
                    topping = " ".join(comp[1:])
            all_topping.append({"NOT": not_flag, "Quantity": quantity, "Topping": topping})
        elif key == 'TOPPING':
            topping = " ".join(attr[1:])
            all_topping.append({"NOT": False, "Quantity": None, "Topping": topping})
        elif key == 'NOT':
            nested = attr[1]
            nested_key = nested[0]
            if nested_key == 'TOPPING':
                topping = " ".join(nested[1:])
                all_topping.append({"NOT": True, "Quantity": None, "Topping": topping})
            elif nested_key == 'STYLE':
                style_text = " ".join(nested[1:])
                style.append({"NOT": True, "TYPE": style_text})
            elif nested_key == 'COMPLEX_TOPPING':
                quantity, topping = None, None
                for comp in nested[1:]:
                    if comp[0] == 'QUANTITY':
                        quantity = " ".join(comp[1:])
                    elif comp[0] == 'TOPPING':
                        topping = " ".join(comp[1:])
                all_topping.append({"NOT": True, "Quantity": quantity, "Topping": topping})

    return {
        "NUMBER": number,
        "SIZE": size,
        "STYLE": style,
        "AllTopping": all_topping
    }


def process_drink_order(attributes):
    # attributes something like:
    # [['NUMBER','five'],['SIZE','large'],['CONTAINERTYPE','bottles'],['DRINKTYPE','coke']]
    # We want:
    # {
    #   "NUMBER": "five",
    #   "SIZE": "large",
    #   "DRINKTYPE": "coke",
    #   "CONTAINERTYPE": "bottles"
    # }

    number = None
    size = None
    drinktype = None
    containertype = None

    for attr in attributes:
        key = attr[0]
        if key == 'NUMBER':
            number = " ".join(attr[1:])
        elif key == 'SIZE':
            size = " ".join(attr[1:])
        elif key == 'DRINKTYPE':
            drinktype = " ".join(attr[1:])
        elif key == 'CONTAINERTYPE':
            containertype = " ".join(attr[1:])

    return {
        "NUMBER": number,
        "SIZE": size if size else None,
        "DRINKTYPE": drinktype,
        "CONTAINERTYPE": containertype
    }



# Let's simplify: We'll read the entire input at once, parse it into one big list and then
# identify top-level orders. Each top-level order seems to be of the form: '(' <ORDERNAME> ... ')'

def parse_and_convert_to_json(input_str):
    # Tokenize and parse once
    tokens = tokenize(input_str)
    # Now, we expect a sequence of top-level forms. Let's parse them all:
    top_level_forms = []
    idx = 0

    def parse_form(idx):
        # expects tokens[idx] == '('
        assert tokens[idx] == '('
        idx += 1
        form = []
        while idx < len(tokens):
            if tokens[idx] == '(':
                subform, idx = parse_form(idx)
                form.append(subform)
            elif tokens[idx] == ')':
                idx += 1
                return form, idx
            else:
                form.append(tokens[idx])
                idx += 1
        return form, idx

    while idx < len(tokens):
        if tokens[idx] == '(':
            form, idx = parse_form(idx)
            top_level_forms.append(form)
        else:
            idx += 1

    # Now we have a list like:
    # [
    #   ['PIZZAORDER', ['NUMBER','two'], ['SIZE','large'], ...],
    #   ['PIZZAORDER', ...],
    #   ['DRINKORDER', ...],
    #   ...
    # ]

    # Convert to desired structure
    result = process_structure(top_level_forms)
    return json.dumps(result, indent=2)
