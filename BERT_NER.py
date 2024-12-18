from Model import *
from transformers import pipeline, AutoModelForTokenClassification

class BERT_NER(Model):
    def __init__(self, model_dir, tokenizer):
        model = AutoModelForTokenClassification.from_pretrained(model_dir)
        self.pipeline = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy="none")

    def predict_labels(self, sentence):
        ner_results = self.pipeline(sentence)
        words, labels = postprocess_results(ner_results)
        return words, labels

def postprocess_results(ner_results):

    words = [ner_result['word'] for ner_result in ner_results]
    predicted_labels = [ner_result['entity'] for ner_result in ner_results]
    start_positions = [ner_result['start'] for ner_result in ner_results]
    end_positions = [ner_result['end'] for ner_result in ner_results]

    # Initial preprocessing to handle apostrophes and subword tokens
    connected_words = []
    connected_labels = []
    connected_starts = []
    connected_ends = []
    j = 0
    while j < len(words):
        if words[j] == "'":
            # Merge apostrophe with the previous word
            if connected_words:
                connected_words[-1] += "'" + words[j+1]
                # Optionally, you can adjust how labels are handled here
                # For simplicity, we'll keep the label of the previous word
                connected_ends[-1] = end_positions[j+1]
            j += 2
        elif words[j] == ":":
            j += 1
        elif words[j].startswith('##'):
            # Merge subword token with the previous word, removing the '##' prefix
            if connected_words:
                connected_words[-1] += words[j][2:]
                connected_ends[-1] = end_positions[j]
            j += 1
        else:
            # Append the current word and its label
            connected_words.append(words[j])
            connected_labels.append(predicted_labels[j])
            connected_starts.append(start_positions[j])
            connected_ends.append(end_positions[j])
            j += 1

    # Post-processing to merge hyphenated words (e.g., 'coca', '-', 'cola') into single tokens
    j = 0
    while j < len(connected_words) - 2:
        word1, hyphen, word2 = connected_words[j], connected_words[j+1], connected_words[j+2]
        start1, hyphen_start, word2_start = connected_starts[j], connected_starts[j+1], connected_starts[j+2]
        end1, hyphen_end, word2_end = connected_ends[j], connected_ends[j+1], connected_ends[j+2]
        
        # Check if the middle token is a hyphen
        if hyphen == '-':
            # Check if hyphen is directly connected to word1 and word2 (no spaces)
            # This means:
            # - hyphen_start should be equal to end1 (no space between word1 and hyphen)
            # - word2_start should be equal to hyphen_end (no space between hyphen and word2)
            if (hyphen_start == end1) and (word2_start == hyphen_end):
                # Merge the words into 'word1-word2'
                merged_word = f"{word1}-{word2}"
                connected_words[j] = merged_word
                # Keep the label of 'word1'
                connected_labels[j] = connected_labels[j]
                # Update the end position
                connected_ends[j] = word2_end
                # Remove the hyphen and word2 from the lists
                del connected_words[j+1:j+3]
                del connected_labels[j+1:j+3]
                del connected_starts[j+1:j+3]
                del connected_ends[j+1:j+3]
                # After merging, stay at the same index to check for overlapping sequences
                continue
        j += 1

    # if connected_words[-1] == 'please': connected_labels[-1] = 'OTHER'

    # for j in range(len(connected_labels)-1):
    #   if connected_labels[j] == 'PIZZA_BEGIN' and connected_labels[j+1] == 'DRINK_INTERMEDIATE': connected_labels[j] = 'DRINK_BEGIN'
    #   if connected_labels[j] == 'DRINK_BEGIN' and connected_labels[j+1] == 'PIZZA_INTERMEDIATE': connected_labels[j] = 'PIZZA_BEGIN'

    # for j in range(len(connected_labels)-1):
    #   if connected_labels[j] == 'PIZZA_INTERMEDIATE' and connected_labels[j+1] == 'PIZZA_BEGIN': connected_labels[j] = 'OTHER'
    #   if connected_labels[j] == 'DRINK_INTERMEDIATE' and connected_labels[j+1] == 'DRINK_BEGIN': connected_labels[j] = 'OTHER'

    # for j in range(len(connected_labels)-1):
    #   if connected_labels[j] == 'PIZZA_BEGIN' and connected_labels[j+1] != 'PIZZA_INTERMEDIATE': connected_labels[j] = 'OTHER'
    #   if connected_labels[j] == 'DRINK_BEGIN' and connected_labels[j+1] != 'DRINK_INTERMEDIATE': connected_labels[j] = 'OTHER'

    stack = []
    for j in range(len(connected_labels)):
        if connected_labels[j] == 'PIZZA_BEGIN':
            stack.append(j)
        elif connected_labels[j] == 'PIZZA_INTERMEDIATE':
            if len(stack) == 0:
                connected_labels[j] = 'PIZZA_BEGIN'
                stack.append(j)
        elif connected_labels[j] == "OTHER":
            if j > 0 and connected_labels[j-1] == "PIZZA_INTERMEDIATE":
                stack.pop()

        if connected_labels[j] == 'DRINK_BEGIN':
            stack.append(j)
        elif connected_labels[j] == 'DRINK_INTERMEDIATE':
            if len(stack) == 0:
                connected_labels[j] = 'DRINK_BEGIN'
                stack.append(j)
        elif connected_labels[j] == "OTHER":
            if j > 0 and connected_labels[j-1] == "DRINK_INTERMEDIATE":
                stack.pop()

    return connected_words, connected_labels