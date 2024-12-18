class Model:
    '''
    An abstract class that outline what each model class should contain. Do not change anything here.
    If you wish to implement a model, inherit from this class and write down your custom implementation. (See BERT_NER.py for example)
    
    Make sure to NOT change the parameters of the generate_labels function. You can do whatever you want with the constructor,
    just don't change anything about the generate_labels
    '''
    def __init__(self, model_path):
        '''
        Load your model and all of its needed dependencies (tokenizer, embeddings, etc) here
        '''
        pass
    

    def generate_labels(self, sentence):
        '''
        This method takes in an input string representing a sentence in the required format, and outputs 2 lists:

        1) A list of words containing every word in the sentence. For example, if the sentence is "i want pizza", then this list will
        be ['i', 'want', 'pizza']. Just use the .spit() function on the sentence.

        2) A list of labels, the outputs the model. It MUST be of the same length as the words list, make sure to preprocess to handle
        any tokenization/padding.
        '''
        pass