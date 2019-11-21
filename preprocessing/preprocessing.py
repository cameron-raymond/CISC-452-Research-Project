import numpy                    as np
import preprocessing_helpers    as preproc
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding
from tensorflow.keras.models import load_model



if __name__ == "__main__":
    text_df, max_length   = preproc.return_data('../data/{}.csv'.format("train"),num_to_take=80,clean=True)
    labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    train  = np.array(text_df['cleaned_text'])
    t = Tokenizer(filters = '"#$%&()*+-/:;<=>@[\]^_`{|}~')
    # generate a vocabulary based on frequency based on the texts
    t.fit_on_texts(train)
    # Generates a matrix where each row is a document
    train = t.texts_to_sequences(train)
    train = pad_sequences(train, maxlen=max_length, padding='post')

    mask = np.random.rand(len(train)) < 0.8
    x_train, x_test = train[mask], train[~mask] # split data into train test splits
    y_train, y_test = np.array(text_df[mask][labels]), np.array(text_df[~mask][labels])
    
    glove_embeddings = preproc.load_glove('../glove/glove.6B.100d.txt')
    embedding_matrix = preproc.generate_glove_weights(glove_embeddings,t,check_exists=False)

    print("--- Initializing Model ---")
    VOCAB_SIZE = embedding_matrix.shape[0]
    dense_model = Sequential()
    e = Embedding(VOCAB_SIZE, 100, weights=[embedding_matrix], input_length=max_length, trainable=False)
    dense_model.add(e)
    dense_model.add(Flatten())
    dense_model.add(Dense(6, activation='relu'))
    # compile the desnse_model
    print("--- Training ---")
    dense_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # summarize the desnse_model
    print(dense_model.summary())
    # fit the desnse_model
    print(type(x_train),type(y_train))
    dense_model.fit(x_train, y_train, epochs=10, verbose=1)

    loss, accuracy = dense_model.evaluate(x_test,y_test, verbose=0)
    print('Accuracy: {:.3f}'.format(accuracy*100))