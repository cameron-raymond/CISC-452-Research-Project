from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding

import preprocessing_glove as preproc
import numpy as np


def load_h5_model(name):
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
    file_path = './saved_models/{}.h5'.format(name)
    if os.path.exists(file_path):
        return load_model(file_path)
    return None

def save_h5_model(model,name="model"):
    """
    Saves in a model to the HDF5 binary data format.

    Parameters
    ----------
    model : tensorflow model 
        The Tensorflow model to be saved

    name : str
        The name of the HDF5 model. Checks if the model exists within the saved_models directory.
    """
    file_path = './saved_models/{}.h5'.format(name)
    if not os.path.exists(os.path.dirname(file_path)):
        try:
            os.makedirs(os.path.dirname(file_path))
            model.save(file_path)
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


def main():
    labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    text_df, max_length = preproc.return_data('./data/cleaned_train.csv')

    x_train = np.array(text_df['cleaned_text'])
    x_labels = np.array(text_df.loc[:][labels])
    
    print("--- Loading embeddings ---")
    embeddings = preproc.load_glove('./glove/glove.6B.100d.txt')
    print("--- Done Loading Embeddings. Now Fitting Vocab ---")

    t = Tokenizer(filters = '"#$%&()*+-/:;<=>@[\]^_`{|}~')
    # generate a vocabulary based on frequency based on the texts
    t.fit_on_texts(x_train)

    # Generates a matrix where each row is a document
    x_train = t.texts_to_sequences(x_train)
    x_train = pad_sequences(x_train, maxlen=max_length, padding='post')
    embedding_matrix = preproc.generate_glove_weights(embeddings, t)



    VOCAB_SIZE = embedding_matrix.shape[0]

    model_name = "glove_embedding_NN"
    model = Sequential()
    e = Embedding(VOCAB_SIZE, 100, weights=[embedding_matrix], input_length=max_length, trainable=False)
    model.add(e)
    model.add(Flatten())
    model.add(Dense(6, activation='relu'))
    # compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # summarize the model
    print(model.summary())
    # fit the model
    model.fit(x_train, x_labels, epochs=10, verbose=1)
    # evaluate the model
    loss, accuracy = model.evaluate(x_train, x_labels, verbose=0)
    print('Accuracy: {:.3f}'.format(accuracy*100))
    print("Saving model to file...")
    save_h5_model(model,model_name)
    del model
    print("Loading model...")
    model = load_h5_model(model_name)
    loss, accuracy = model.evaluate(x_train, x_labels, verbose=0)
    print('Accuracy: {:.3f}'.format(accuracy*100))

main()
