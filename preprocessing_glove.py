import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
import csv

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
# nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
# from tensorflow.keras.layers import Dense, Flatten, Embedding

# How many words to consider
VOCAB_SIZE = 1000000




def text_clean(text):
    """
    Gets and prints the spreadsheet's header columns

    Parameters
    ----------
    text : str
        Any sentence string

    Returns
    -------
    list
        a list of strings that are lemmatized words without stop words
    """
    lemmatize   = lambda token_list : \
                        [ lemmatizer.lemmatize(token) for token in token_list \
                        if token not in stopwords.words('english') and len(token) > 3] 
                        
    text        = text.lower()
    token_words = word_tokenize(text)
    return lemmatize(token_words)



# TODO
#   see if max_length is correct
def return_data(path):
    """
    Returns the csv's comments and their associated labels

    Parameters
    ----------
    path : str
        String path to the csv containing the training or test data

    Returns
    -------
    docs
        a list of strings with each one being a comment from a different document
    labels
        a list of 0 or 1s that represent the label
    """
    with open(path, encoding="utf8") as csv_file:
        csv_reader = csv.reader(csv_file)
        docs = []
        labels = []
        line_count = 0
        max_length = 0
        for line in csv_reader:
            cleaned_text = " ".join(text_clean(line[1]))
            curr_length = len(text_clean(line[1]))
            print(cleaned_text)
            print(curr_length)
            print("\n")
            docs.append(cleaned_text)
            labels.append(line[2:])
            if curr_length > max_length:
                max_length = curr_length
            if line_count == 1000:
                break
            line_count += 1
        return docs, labels, max_length

def load_glove(path):
    """
    Loads in the glove embedding from a text file

    Parameters
    ----------
    path : str
        String path to the txt file containing the glove data

    Returns
    -------
    embeddings_index : dict
        A dictionary with keys being words and values are the vector embeddings
    """
    embeddings_index = dict()
    f = open(path , encoding='utf8')
    for line in f:
        values = line.split()
        word = values[0]
        coeffs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coeffs
    f.close()
    return embeddings_index


def generate_glove_weights(embeddings, tokenizer):
    """
    generates a weight matrix to be input in an embedding layer in keras

    Returns
    -------
    embeddings_matrix : np.array
        A numpy array of dimensions vocab_size x vector_representation_dim. Each row represents a word that appears
        during in a document and the vectors can be thought of as the weights being used in training.
    """
    embedding_matrix = np.zeros((VOCAB_SIZE, 100))
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

def main():
    x_train, y_train, max_length = return_data('./data/train.csv')

    print("Loading glove embeddings")
    embeddings = load_glove('./glove/glove.6B.100d.txt')
    print("Done. Now fitting vocab...")


    t = Tokenizer(num_words = VOCAB_SIZE, filters = '"#$%&()*+-/:;<=>@[\]^_`{|}~')
    # generate a vocabulary based on frequency based on the texts
    t.fit_on_texts(x_train)

    # Generates a matrix where each row is a document
    x_train = t.texts_to_sequences(x_train)
    x_train = pad_sequences(x_train, maxlen=max_length, padding='post')
    glove_matrix = generate_glove_weights(embeddings, t)

    print(glove_matrix[1])
    print(glove_matrix[2])
    print(max_length)


main()