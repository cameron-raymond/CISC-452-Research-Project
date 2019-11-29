import numpy                    as np
import preprocessing.preprocessing_helpers    as preproc
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding, Conv1D, GlobalMaxPooling1D, Dropout
from tensorflow.keras.models import load_model

from tensorflow.keras import backend as K

def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

if __name__ == "__main__":
    text_df, max_length   = preproc.return_data('./data/{}.csv'.format("balanced_train"),clean=False)
    labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    # testing stuff
    test_df, test_max_length = preproc.return_data('./data/cleaned_test_labels.csv')
    test_data = np.array(test_df['cleaned_text'])
    test_labels = np.array(test_df[labels])



    train  = np.array(text_df['cleaned_text'])
    t = Tokenizer(filters = '"#$%&()*+-/:;<=>@[\]^_`{|}~')
    # generate a vocabulary based on frequency based on the texts
    t.fit_on_texts(train)
    # Generates a matrix where each row is a document
    train = t.texts_to_sequences(train)
    train = pad_sequences(train, maxlen=max_length, padding='post')
    

    # test Tokenizer stuff
    test_data = t.texts_to_sequences(test_data)
    test_data = pad_sequences(test_data, maxlen=test_max_length, padding='post')


    mask = np.random.rand(len(train)) < 0.8
    x_train, x_test = train[mask], train[~mask] # split data into train test splits
    y_train, y_test = np.array(text_df[mask][labels]), np.array(text_df[~mask][labels])
    
    glove_embeddings = preproc.load_glove('./glove/glove.6B.100d.txt')
    embedding_matrix = preproc.generate_glove_weights(glove_embeddings,t,check_exists=False)

    print("--- Initializing Model ---")
    VOCAB_SIZE = embedding_matrix.shape[0]
    model = Sequential()
    e = Embedding(VOCAB_SIZE, 100, weights=[embedding_matrix], input_length=max_length, trainable=False)
    model.add(e)
    model.add(Conv1D(100, 5, activation='relu'))
    #model.add(Dropout(0.1))
    model.add(Conv1D(50,5, activation='relu'))
    #model.add(Dropout(0.1))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(6, activation='sigmoid'))
    # compile the desnse_model
    print("--- Training ---")

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy',f1_m,precision_m, recall_m])
    # summarize the desnse_model
    print(model.summary())
    # fit the desnse_model
    print(type(x_train),type(y_train))
    model.fit(x_train, y_train, epochs=5, verbose=1)

    #loss, accuracy = model.evaluate(x_test,y_test, verbose=1)
    #print('Accuracy: {:.3f}'.format(accuracy*100))

    loss, accuracy, f1_score, precision, recall = model.evaluate(test_data, test_labels, verbose=1)
    print('Accuracy: {:.3f}'.format(accuracy*100))
    print('Loss:',loss)
    print('Fscore:',f1_score)
    print('Precision:',precision)
    print('Recall:',recall)
    
  