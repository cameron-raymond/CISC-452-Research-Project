import numpy as np
import pandas as pd
import preprocessing.preprocessing_helpers as preproc
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, LSTM, Flatten, Embedding, SpatialDropout1D,GlobalMaxPool1D, Dropout
from tensorflow.keras.models import load_model, save_model

class Dense_Toxic_Comment(object):
    def __init__(self,x_train=None,y_train=None,x_test=None,y_test=None,embedding_matrix=None,max_length=None,saved_model=None):
        super().__init__()
        self.max_length     = max_length
        self.x_train        = x_train
        self.y_train        = y_train
        self.x_test         = x_test
        self.y_test         = y_test
        self.model          = self.define_model(embedding_matrix,saved_model=saved_model)

    def define_model(self,embedding_matrix,saved_model=None):
        if saved_model is None:
            print("--- Initializing Model ---")
            num_labels = self.y_train.shape[1]
            VOCAB_SIZE = embedding_matrix.shape[0]
            model = Sequential()
            embedding_layer = Embedding(VOCAB_SIZE, 100, weights=[embedding_matrix], input_length=max_length, trainable=False)
            model.add(embedding_layer)
            model.add(Flatten())
            model.add(Dense(32, activation="relu"))
            model.add(Dropout(0.2))
            model.add(Dense(16, activation="relu"))
            model.add(Dropout(0.2))
            model.add(Dense(num_labels, activation="sigmoid"))
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
        epochs = 5
        batch_size = 64
        callback = EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001,verbose=1)
        history = self.model.fit(x_train,y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1,callbacks=[callback],verbose=1)

    def validate(self):
        # Evaluate the model on the test data using `evaluate`
        print('\n# Evaluate on test data')
        print(self.x_test.shape,self.y_test.shape)
        results = self.model.evaluate(self.x_test, self.y_test, batch_size=128,verbose=0)
        print('test loss, test acc:', results)

    def save_model(self,file_path="saved_models/dense_model.h5"):
        save_model(self.model, file_path)
        print("--- model saved to {} ---".format(file_path))

if __name__ == "__main__":
    train_df, max_length   = preproc.return_data('./data/{}.csv'.format("balanced_train"))
    labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']    
    train  = np.array(train_df['cleaned_text'])
    t = Tokenizer(filters = '"#$%&()*+-/:;<=>@[\]^_`{|}~')

    # generate a vocabulary based on frequency based on the texts
    t.fit_on_texts(train)
    print("--- generating GloVe embedding layer --")
    glove_embeddings = preproc.load_glove('./glove/glove.6B.100d.txt')
    embedding_matrix = preproc.generate_glove_weights(glove_embeddings,t,check_exists=False)
    print("\t* Embedding Matrix Dimensions: {}".format(embedding_matrix.shape))

    # Generates a matrix where each row is a document
    train = t.texts_to_sequences(train)
    train = pad_sequences(train, maxlen=max_length, padding='post')
    mask = np.random.rand(len(train)) < 0.8
    x_train, x_test = train[mask], train[~mask] # split data into train test splits
    y_train, y_test = np.array(train_df[mask][labels]), np.array(train_df[~mask][labels])
    print(x_test.shape)

    # Train and/or validate model
    using_saved_model = False  # Change this to True to only evaluate model on test data
    if using_saved+_model:
        saved_dense = Dense_Toxic_Comment(x_test=x_test,y_test=y_test,saved_model="saved_models/dense_model.h5")
        saved_dense.validate()
    else:
        dense_toxic_Comment = Dense_Toxic_Comment(x_train,y_train,x_test,y_test,embedding_matrix,max_length)
        dense_toxic_Comment.train()
        dense_toxic_Comment.save_model()
        dense_toxic_Comment.validate()
