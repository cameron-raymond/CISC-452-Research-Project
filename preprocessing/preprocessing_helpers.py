import numpy as np
import os.path
import csv
import pandas as pd
import sys
import nltk
import h5py # for saving models to file
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
from tensorflow.keras.models import load_model as tf_load
# nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()



from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding

ROOT = os.path.dirname(os.path.dirname(__file__))

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
    return " ".join(lemmatize(token_words))



def return_data(path,num_to_take=None,clean=False,write=False):
    """
    Reads in a csv turns the csv's comments as a data frame along with the largest length 
    Parameters
    ----------
    path : str
        String path to the csv containing the training or test data
    num_to_take : None | int
        If num_to_take is None then return the entire data set, otherwise return the number of rows equal to num_to_take
    Returns
    -------
    text_df : pandas.DataFrame
        a pandas DataFrame containing the entire document with cleaned text and labels.
    max_len: int
        an integer representing the number of words in the longest comment in the cleaned_text column
    """

    list_len = lambda p : len(p.split())
    print("--- reading in first {} from CSV ---".format(num_to_take)) if num_to_take else print("--- reading in CSV ---")
    text_df = pd.read_csv(path, nrows=num_to_take) if num_to_take else pd.read_csv(path)
    if clean:
        text_df["cleaned_text"] = text_df["comment_text"].apply(text_clean)
    text_df["cleaned_text"] = text_df["cleaned_text"].astype(str)
    max_len = max(text_df['cleaned_text'].apply(list_len))
    return text_df,max_len

def clean_all_data(path):
    """
    takes in a csv file and cleans all the comment_text column. This is only used to clean the train.csv and test.csv
    Parameters
    ----------
    path : str
        String path to the csv containing the training or test data
    """
    print("--- reading in CSV ---")
    text_df = pd.read_csv(path)
    print("--- cleaning ---")
    text_df['cleaned_text'] = text_df['comment_text'].apply(text_clean)
    text_df.to_csv('./data/cleaned_text.csv', encoding='utf-8')
    print("--- Cleaned all data! ---")


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

def generate_glove_weights(embeddings, tokenizer,check_exists=False):
    """
    generates a weight matrix. It is then saved as a binary numpy file that can be loaded
    Generates
    ---------
    embedding_matrix.npy : np.array
        A numpy array of dimensions vocab_size x vector_representation_dim. Each row represents a word that appears
        during in a document and the vectors can be thought of as the weights being used in training.
    """
    if os.path.exists("./data/embedding_matrix.npy") and check_exists:
        print("--- loading GloVe weights ---")
        return np.load("./data/embedding_matrix.npy")
    print("--- generating GloVe weights ---")
    VOCAB_SIZE = len(tokenizer.word_index) + 1
    embedding_matrix = np.zeros((VOCAB_SIZE,100))
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    np.save('{}/data/embedding_matrix.npy'.format(ROOT), embedding_matrix)
    print('--- Saved embedding matrix ---')
    return embedding_matrix

def load_model(name):
    """
    Loads in a model saved with the HDF5 binary data format.
    Parameters
    ----------
    name : str
        The name of the HDF5 model. Checks if the model exists within the saved_models directory.
    Returns
    -------
    model : tensorflow model | None
    """
    file_path = '{}/saved_models/{}.h5'.format(ROOT,name)
    if os.path.exists(file_path):
        return tf_load(file_path)
    return None

def confusion_matrix(y_true,y_pred,labels):
    conf_mat_dict={}

    for label_col in range(len(labels)):
        y_true_label = y_true[:, label_col]
        y_pred_label = y_pred[:, label_col]
        conf_mat_dict[labels[label_col]] = confusion_matrix(y_pred=y_pred_label, y_true=y_true_label)


    for label, matrix in conf_mat_dict.items():
        print("Confusion matrix for label {}:".format(label))
        print(matrix)
    return conf_mat_dict

def to_binary(data,labels):
    print("--- converting multi label problem to binary classification --- ")
    data["any_toxicity"] = data[labels].apply(lambda row: (row == 1).any() , axis=1)*1 # The .any() checks to see if any of the labels are present, *1 casts it to an int (false == 0, true == 1)
    binary_data = data.drop(labels,axis = 1)
    return binary_data
    
def load_h5_model(file_path):
    """
    Loads in a model saved with the HDF5 binary data format.
    Parameters
    ----------
    file_path : str
        The file_path of the HDF5 model. Checks if the model exists within the saved_models directory.
    Returns
    -------
    model : tensorflow model | None
    """
    file_path = "{}/{}".format(ROOT,file_path)
    if os.path.exists(file_path):
        return tf_load(file_path)
    return None

def save_model(name="model"):
    """
    Saves in a model to the HDF5 binary data format.
    Parameters
    ----------
    name : str
        The name of the HDF5 model. Checks if the model exists within the saved_models directory.
    """
    file_path = '{}/saved_models/{}.h5'.format(ROOT,name)
    print("model => {}".format(file_path))
    
    if not os.path.exists(os.path.dirname(file_path)):
        try:
            os.makedirs(os.path.dirname(file_path))
            self.to_dense.save(file_path)
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    else:
        self.to_dense.save(file_path)