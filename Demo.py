from LSTM import Toxic_Comment_LSTM
import time
import sys
import numpy as np
import pandas as pd
import preprocessing.preprocessing_helpers as preproc
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model


def validate(model,x_test,y_test,numToValidate=None):
        # Evaluate the model on the test data using `evaluate`
        if numToValidate and numToValidate <= x_test.shape[0]-1:
            x_test = x_test[:numToValidate]
            y_test = y_test[:numToValidate]
        print('--- Evaluate on test data ---')
        print("\t* Size: {}, num labels: {}".format(x_test.shape,y_test.shape))
        results = model.evaluate(x_test, y_test, batch_size=128,verbose=0)
        print('\t* test loss, test acc:', results)
        # Generate predictions (probabilities -- the output of the last layer)
        # on new data using `predict`
        shuffle_idx = np.random.choice(np.arange(x_test.shape[0]), 50, replace=False)
        x_sample, y_sample = x_test[shuffle_idx],y_test[shuffle_idx]
        print('\n--- Generate predictions for {} samples ---'.format(x_sample.shape[0]))
        predictions = model.predict(x_sample).round()
        num_correct = (predictions == y_sample).all(1).sum()
        print("--- Num correct from sample {}. ({:.2f}%) ---".format(num_correct,num_correct/len(predictions)))
        for i,(prediction,target) in enumerate(zip(predictions,y_sample)):
            print("\t* prediction {} -> {}".format(i,prediction))
            print("\t\t* targ vec -> {}".format(target))
        print('predictions shape:', predictions.shape)



if __name__ == "__main__":
    
    numToValidate = int(sys.argv[1]) if len(sys.argv) > 1 and  sys.argv[1].isdigit() else 1000
    start = time.time()
    model_str = "./saved_models/{}.h5"
    print("--- Import training data (used for tokenizing) ---")
    train_df, max_length   = preproc.return_data('./data/{}.csv'.format("cleaned_train"))
    labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']    
    train  = np.array(train_df['cleaned_text'])
    t = Tokenizer(filters = '"#$%&()*+-/:;<=>@[\]^_`{|}~')
    # generate a vocabulary based on frequency based on the texts
    print("--- Fit training data to tokenizer ---")
    t.fit_on_texts(train) # fit tokenizer
    # Generates a matrix where each row is a document
    print("--- import test data ---")
    test_df, _ = preproc.return_data('./data/{}.csv'.format("cleaned_test"))
    test_df = np.array(test_df["cleaned_text"])
    print("--- convert each document to frequency vector --")
    x_test = t.texts_to_sequences(test_df)
    x_test = pad_sequences(x_test, maxlen=max_length, padding='post')
    print("--- import test labels ---")
    y_test = pd.read_csv('./data/{}.csv'.format("test_labels"))
    y_test = y_test[labels].to_numpy()
    print("--- loading in LSTM ---")
    LSTM = load_model(model_str.format("toxic_comment_LSTM"))
    print("--- Validating on {} test documents ---".format(numToValidate))
    validate(LSTM,x_test,y_test,numToValidate)
