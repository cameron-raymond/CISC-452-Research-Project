import numpy as np
import pandas as pd
import preprocessing.preprocessing_helpers as preproc
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, LSTM, Flatten, Embedding, SpatialDropout1D,GlobalMaxPool1D, Dropout
from tensorflow.keras.models import load_model, save_model

VOCAB_SIZE = 5000


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
        model = preproc.load_model("dense_model")
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

        # Generate predictions (probabilities -- the output of the last layer)
        # on new data using `predict`
        # shuffle_idx = np.random.choice(np.arange(self.x_test.shape[0]), 50, replace=False)
        # x_sample, y_sample = self.x_test[shuffle_idx],self.y_test[shuffle_idx]
        # print('\n# Generate predictions for {} samples'.format(x_sample.shape[0]))
        # predictions = self.model.predict(x_sample).round()
        # num_correct = (predictions == y_sample).all(1).sum()
        # print("Num correct from sample {}. ({:.2f}%)".format(num_correct,num_correct/len(predictions)))
        # for i,(prediction,target) in enumerate(zip(predictions,y_sample)):
        #     print("\t* prediction {} -> {}".format(i,prediction))
        #     print("\t\t*targ vec -> {}".format(target))
        # print('predictions shape:', predictions.shape)

    def save_model(self,file_path="saved_models/dense_model.h5"):
        save_model(self.model, file_path)
        print("--- model saved to {} ---".format(file_path))

# Testing saved model
# if __name__ == "__main__":
#     labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']    
#     train_df, max_length = preproc.return_data('data/{}.csv'.format("balanced_train"))
#     t   = Tokenizer(filters = '"#$%&()*+-/:;<=>@[\]^_`{|}~')
#     t.fit_on_texts(train_df['cleaned_text'])

#     test_df, _ = preproc.return_data('data/{}.csv'.format("cleaned_test"))
#     x_test = np.array(test_df['cleaned_text'])
#     x_test = t.texts_to_sequences(x_test)
#     x_test = pad_sequences(x_test, maxlen=max_length, padding='post')
#     y_test = pd.read_csv("data/test_labels.csv")
#     y_test = np.array(y_test[labels])
#     saved_dense = Dense_Toxic_Comment(x_test=x_test,y_test=y_test,saved_model="saved_models/dense_toxic_comment.h5")
#     saved_dense.validate()

# Training new model
if __name__ == "__main__":
    train_df, max_length   = preproc.return_data('./data/{}.csv'.format("balanced_train"), num_to_take=40000)
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
    dense_toxic_Comment = Dense_Toxic_Comment(x_train,y_train,x_test,y_test,embedding_matrix,max_length)
    dense_toxic_Comment.train()
    dense_toxic_Comment.save_model()
    dense_toxic_Comment.validate()
