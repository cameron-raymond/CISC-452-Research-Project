import numpy as np
import csv
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
# nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()



from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding


# How many words to consider
VOCAB_SIZE = 5000

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



def return_data(path):
    """
    Reads in a csv turns the csv's comments as a data frame along with the largest length 

    Parameters
    ----------
    path : str
        String path to the csv containing the training or test data

    Returns
    -------
    text_df : pandas.DataFrame
        a pandas DataFrame containing the entire document with cleaned text and labels.

    max_len: int
        an integer representing the number of words in the longest comment in the cleaned_text column
    """

    list_len = lambda p : len(p.split())
    print("--- reading in CSV ---")
    text_df = pd.read_csv(path)
    text_df["cleaned_text"] = text_df["cleaned_text"].astype(str)
    max_len = max(text_df['cleaned_text'].apply(list_len))

    return text_df,max_len

def clean_all_data(path):
    """
    takes in a csv file and cleans all the comment_text column

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



def generate_glove_weights(embeddings, tokenizer):
    """
    generates a weight matrix to be input in an embedding layer in keras

    Returns
    -------
    embeddings_matrix : np.array
        A numpy array of dimensions vocab_size x vector_representation_dim. Each row represents a word that appears
        during in a document and the vectors can be thought of as the weights being used in training.
    """
    VOCAB_SIZE = len(tokenizer.word_index) + 1
    embedding_matrix = np.zeros((VOCAB_SIZE, 100))
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def testing():
    text_df, max_length = return_data('./data/cleaned_train.csv')


    labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    x_train = text_df['cleaned_text']
    x_labels = text_df.loc[:][labels]

    print("Loading glove embeddings")
    embeddings = load_glove('./glove/glove.6B.100d.txt')
    print("Done. Now fitting vocab...")
 

    t = Tokenizer(filters = '"#$%&()*+-/:;<=>@[\]^_`{|}~')
    # generate a vocabulary based on frequency based on the texts
    t.fit_on_texts(x_train)

    # Generates a matrix where each row is a document
    x_train = t.texts_to_sequences(x_train)
    x_train = pad_sequences(x_train, maxlen=max_length, padding='post')
    embedding_matrix = generate_glove_weights(embeddings, t)


    VOCAB_SIZE = len(t.word_index) + 1

    # Define the model
    model = Sequential()
    e = Embedding(VOCAB_SIZE, 100, weights=[embedding_matrix], input_length=max_length, trainable=False)
    model.add(e)
    model.add(Flatten())
    model.add(Dense(6, activation='sigmoid'))
    # compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # summarize the model
    print(model.summary())
    # fit the model
    model.fit(x_train, x_labels, epochs=10, verbose=0)
    # evaluate the model
    loss, accuracy = model.evaluate(x_train, x_labels, verbose=0)
    print('Accuracy: %f' % (accuracy*100))

testing()