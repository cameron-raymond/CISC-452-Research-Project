from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding
from tensorflow.keras.models import load_model

import os.path
import preprocessing_helpers as preproc
import numpy as np

class Preprocessor(object):
    """
    Encompasses all of the logic needed for the preprocessing workflow.

    Steps include:
    1. Vectorizing data: List of tokens -> n dimensional sparse vector
    2. Training a document embedding model using GloVe  

        * GloVe is an unsupervised learning algorithm for obtaining vector representations for words.
        * This is represented as a dense, feedforward network that would taken in the sparse vector (repsenting the document) and convert it to
            a 100 dimensional dense vector. 

    Parameters
    ----------
    data_name : str
        A string which the preprocessor will find the corresponding file for representing the original data
    
    embedding_mat_name : str | None
        A string, the name of the glove document embedding matrix. Checks if the model exists within the data directory; otherwise it generates the embedding matrix based on the glove embeddings. 

    dense_model_name : str | None
        A string, the name of the feedforward dense model that converts the sparse vector represntation to the dense GloVe vector representation.
        Generates the model if it doesn't already exist.
    """
    def __init__(self,data_name="cleaned_train",embedding_mat_name=None,dense_model_name=None,clean=False):
        super().__init__()
        self.text_df, self.max_length   = preproc.return_data('../data/{}.csv'.format(data_name))
        self.labels     = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        self.x_train,t  = self.__init_x_train()
        self.y_train    = np.array(self.text_df.loc[:][self.labels])
        self.embedding_matrix  = self.__init_embedding(embedding_mat_name,t)    
        self.to_dense   = self.fit_dense_model(dense_model_name) # I feel like this could be a better named

    def __init_x_train(self):
        x_train = np.array(self.text_df['cleaned_text'])
        t = Tokenizer(filters = '"#$%&()*+-/:;<=>@[\]^_`{|}~')
        # generate a vocabulary based on frequency based on the texts
        t.fit_on_texts(x_train)
        # Generates a matrix where each row is a document
        x_train = t.texts_to_sequences(x_train)
        x_train = pad_sequences(x_train, maxlen=self.max_length, padding='post')
        return x_train,t

    def fit_dense_model(self,dense_model_name):
        dense_model = None
        if dense_model_name is None:
            print("--- Initializing Model ---")
            VOCAB_SIZE = self.embedding_matrix.shape[0]
            dense_model = Sequential()
            e = Embedding(VOCAB_SIZE, 100, weights=[self.embedding_matrix], input_length=self.max_length, trainable=False)
            dense_model.add(e)
            dense_model.add(Flatten())
            dense_model.add(Dense(6, activation='relu'))
            # compile the desnse_model
            print("--- Training ---")
            dense_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            # summarize the desnse_model
            print(dense_model.summary())
            # fit the desnse_model
            dense_model.fit(self.x_train, self.y_train, epochs=10, verbose=1)
        else:
            print("--- Loading model ---")
            dense_model = self.load_model(dense_model_name)
        loss, accuracy = dense_model.evaluate(self.x_train, self.y_train, verbose=0)
        print('Accuracy: {:.3f}'.format(accuracy*100))
        return dense_model

    def __init_embedding(self,glove_model,tokenizer=None):
        """
        Returns a weight matrix.

        Parameters
        ----------
        name : str
            The name of the glove document embedding matrix. Checks if the model exists within the saved_models directory.

        Returns
        ---------
        embedding_matrix.npy : np.array
            A numpy array of dimensions vocab_size x vector_representation_dim. Each row represents a word that appears
            during in a document and the vectors can be thought of as the weights being used in training.
        """
        if glove_model is None:
            print("--- generating glove weights ---")
            glove_embeddings = preproc.load_glove('../glove/glove.6B.100d.txt')
            embedding_matrix = preproc.generate_glove_weights(glove_embeddings,tokenizer)
            return embedding_matrix
        else:
            print("--- loading glove weights ---")
            return np.load('../data/{}.npy'.format(glove_model))




if __name__ == "__main__":
    preprocessor = Preprocessor()

