import numpy                    as np
import preprocessing.preprocessing_helpers as preproc
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, LSTM, Flatten, Embedding, SpatialDropout1D
from tensorflow.keras.models import load_model

VOCAB_SIZE = 5000

class Toxic_Comment_LSTM(object):
    def __init__(self,x_train,y_train,x_test,y_test,embedding_matrix,max_length):
        super().__init__()
        self.max_length     = max_length
        self.__VOCAB_SIZE   = embedding_matrix.shape[0]
        self.model          = self.define_model(embedding_matrix)
        self.x_train        = x_train
        self.y_train        = y_train
        self.x_test         = x_test
        self.y_test         = y_test
        print(self.y_test[80])

    def define_model(self,embedding_matrix):
        print("--- Initializing Model ---")
        model = Sequential()
        embedding_layer = Embedding(self.__VOCAB_SIZE, 100, weights=[embedding_matrix], input_length=max_length, trainable=False)
        model.add(embedding_layer)
        model.add(SpatialDropout1D(0.2))
        model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(6, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
  
    def train(self):
        epochs = 5
        batch_size = 64
        shuffle_idx = np.random.choice(np.arange(self.x_train.shape[0]), self.x_train.shape[0], replace=False)
        callback = EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001,verbose=1)
        history = self.model.fit(self.x_train,self.y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1,callbacks=[callback],verbose=1)

    def validate(self):
        # Evaluate the model on the test data using `evaluate`
        print('\n# Evaluate on test data')
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

    def save_model(self):
        preproc.save_model('saved_models/toxic_comment_LSTM.h5',self.model)
        print("--- model saved ---")

if __name__ == "__main__":
    text_df, max_length   = preproc.return_data('./data/{}.csv'.format("cleaned_train"),num_to_take=10000)
    labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    train  = np.array(text_df['cleaned_text'])
    t = Tokenizer(filters = '"#$%&()*+-/:;<=>@[\]^_`{|}~')
    # generate a vocabulary based on frequency based on the texts
    t.fit_on_texts(train)
    print("--- generating GloVe embedding layer --")
    glove_embeddings = preproc.load_glove('./glove/glove.6B.100d.txt')
    embedding_matrix = preproc.generate_glove_weights(glove_embeddings,t,check_exists=False)
    # Generates a matrix where each row is a document
    train = t.texts_to_sequences(train)
    train = pad_sequences(train, maxlen=max_length, padding='post')

    mask = np.random.rand(len(train)) < 0.8
    x_train, x_test = train[mask], train[~mask] # split data into train test splits
    y_train, y_test = np.array(text_df[mask][labels]), np.array(text_df[~mask][labels])

    toxic_Comment_LSTM = Toxic_Comment_LSTM(x_train,y_train,x_test,y_test,embedding_matrix,max_length)
    toxic_Comment_LSTM.train()
    toxic_Comment_LSTM.save_model()
    toxic_Comment_LSTM.validate()