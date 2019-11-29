import time
import sys
import os
from LSTM import Toxic_Comment_LSTM
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
        print('\t* test loss: {:.4f}, test acc: {:.4f}'.format(*results))
        print("\n")

        
def return_models(directory):
    models = []
    directory = os.fsencode(directory).decode("utf-8") 
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".h5"): 
            full_path = os.path.join(directory, filename)
            model = load_model(full_path)
            models.append((filename,model))
        else:
            print("Only keep h5 models in the saved models directory")
    return models



if __name__ == "__main__":
    print("--- starting demo ---")
    numToValidate = int(sys.argv[1]) if len(sys.argv) > 1 and  sys.argv[1].isdigit() else 1000
    start = time.time()
    model_str = "./saved_models/"
    models = return_models(model_str)
    print("--- imported {} model(s) ---".format(len(models)))
    print("--- Import training data (used for tokenizing) ---")
    train_df, max_length   = preproc.return_data('./data/{}.csv'.format("balanced_train"))
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
    y_test = pd.read_csv('./data/{}.csv'.format("cleaned_test_labels"))
    y_test = y_test[labels].to_numpy()
    print("--- Validating on {} test documents ---".format(numToValidate))
    for model_name, model in models:
        print("--- Validating {} ---".format(model_name))
        validate(model,x_test,y_test,numToValidate)
    end = time.time()
    print("Total time for validation (seconds): {:.2f}".format(end-start))
