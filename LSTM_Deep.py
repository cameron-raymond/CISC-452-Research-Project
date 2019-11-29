import numpy                    as np
import preprocessing.preprocessing_helpers as preproc
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, LSTM, Flatten, Embedding, SpatialDropout1D, GlobalMaxPool1D, Bidirectional, Dropout
from tensorflow.keras.models import load_model
import tensorflow as tf
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score



class Toxic_Comment_LSTM(object):
    def __init__(self,x_train=None,y_train=None,x_test=None,y_test=None,embedding_matrix=None,max_length=None,saved_model=None):
        super().__init__()
        self.max_length     = max_length
        self.model          = self.define_model(embedding_matrix,saved_model=saved_model)
        self.x_train        = x_train
        self.y_train        = y_train
        self.x_test         = x_test
        self.y_test         = y_test


    # More complicated LSTM with additional layer
    def define_model(self,embedding_matrix,saved_model=None):
        if saved_model is None:
            print("--- Initializing Model ---")
            VOCAB_SIZE = embedding_matrix.shape[0]
            model = Sequential()
            embedding_layer = Embedding(VOCAB_SIZE, 100, weights=[embedding_matrix], input_length=max_length, trainable=False)
            model.add(embedding_layer)
            model.add(SpatialDropout1D(0.2))
            model.add(Bidirectional(LSTM(50, dropout=0.2, recurrent_dropout=0.2)))
            model.add(Dense(50, activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(6, activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            return model
        print("--- Loading Model from {} ---".format(saved_model))
        model = preproc.load_h5_model(saved_model)
        if model is None: # If the filepath is wrong or the model hasn't actually been defined earlier
            print("--- no model found, initializing from scrach ---")
            return self.define_model(embedding_matrix,saved_model=None)
        return model
  
    def train(self,x_train=None,y_train=None):
        x_train = self.x_train if x_train is None else x_train
        y_train = self.y_train if y_train is None else y_train
        epochs = 2
        batch_size = 64
        # callback = EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001,verbose=1)
        history = self.model.fit(x_train,y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1,verbose=1)

    def validate(self):
        # Evaluate the model on the test data using `evaluate`
        print('\n# Evaluate on test data')
        print(self.x_test.shape,self.y_test.shape)
        results = self.model.evaluate(self.x_test, self.y_test, batch_size=128,verbose=0)
        print('test loss, test acc:', results)

        # Generate predictions (probabilities -- the output of the last layer)
        # on new data using `predict`
        shuffle_idx = np.random.choice(np.arange(self.x_test.shape[0]), 10, replace=False)
        x_sample, y_sample = self.x_test[shuffle_idx],self.y_test[shuffle_idx]
        print('\n# Generate predictions for {} samples'.format(x_sample.shape[0]))
        predictions = self.model.predict(x_sample)
        for i,(prediction,target) in enumerate(zip(predictions,y_sample)):
            chosen = np.argmax(prediction)
            actual = np.argmax(target)
            print("\t* prediction {} -> {}, actual -> {}".format(i,chosen,actual))
            print("\t\t* {}".format(actual))
        print('predictions shape:', predictions.shape)

    def save_model(self,file_path="saved_models/toxic_comment_LSTM.h5"):
        
        preproc.save_h5_model(file_path,self.model)
        print("--- model saved to {} ---".format(file_path))

if __name__ == "__main__":
    labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    print(os.getcwd())
    preds = np.loadtxt('./deep_predictions_int.txt', encoding='utf-8')
    preds = preds.astype(int)
    test_df, max_length = preproc.return_data('./data/cleaned_test_labels.csv')
    test_labels = np.array(test_df[labels])


    recall = recall_score(y_true=test_labels, y_pred=preds, average='weighted')
    precision = precision_score(y_true=test_labels, y_pred=preds, average='weighted')
    f_score = f1_score(y_true=test_labels, y_pred=preds, average='weighted')

    print("Recall: ", recall)
    print("precision: ", precision)
    print("f1_score: ", f_score)



    print('--- Reading test data ---')
    test_df, test_max_length = preproc.return_data('./data/cleaned_test_labels.csv')
    test_data = np.array(test_df['cleaned_text'])
    test_labels = np.array(test_df[labels])

    text_df, max_length   = preproc.return_data('./data/{}.csv'.format("upsampled_downsampled_train"))
    train  = np.array(text_df['cleaned_text'])
    t = Tokenizer(filters = '"#$%&()*+-/:;<=>@[\]^_`{|}~')
    t.fit_on_texts(train)
    train = t.texts_to_sequences(train)
    train = pad_sequences(train, maxlen=max_length, padding='post')


    # tokenize the test data
    test_data = t.texts_to_sequences(test_data)
    test_data = pad_sequences(test_data, maxlen=test_max_length, padding='post')



    embedding_matrix = np.load('./data/embedding_matrix.npy')

    mask = np.random.rand(len(train)) < 0.8

    x_train, x_test = train[mask], train[~mask] # split data into train test splits
    y_train, y_test = np.array(text_df[mask][labels]), np.array(text_df[~mask][labels])
    toxic_Comment_LSTM = Toxic_Comment_LSTM(x_test=x_test,y_test=y_test, embedding_matrix=embedding_matrix, saved_model='saved_models/Deep_Model_3_epochs.h5')
    print('--- Predicting ---')
    preds = toxic_Comment_LSTM.model.predict(test_data)
    preds[preds>=0.5] = int(1)
    preds[preds<0.5] = int(0)
    preds = preds.astype(int)
    np.savetxt('./deep_predictions.txt', preds)
    # print('--- saved ---')