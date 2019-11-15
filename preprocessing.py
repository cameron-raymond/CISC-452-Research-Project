#!/usr/local/bin/python3
# import preprocessor as p
import sys
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

def text_clean(text):
    lemmatize   = lambda token_list : \
                        [ lemmatizer.lemmatize(token) for token in token_list \
                        if token not in stopwords.words('english') and len(token) > 3] 
    text        = text.lower()
    token_words = word_tokenize(text)
    return lemmatize(token_words)


if __name__ == "__main__":
    file_name = "train" if len(sys.argv) <=1 else sys.argv[1] #User didn't enter filename from command line
       
    # IMPORT DATA
    data = pd.read_csv("./data/{}.csv".format(file_name)).head(10)
    print("---\tread in {}.csv, Size: {}, number of attributes: {} ---".format(file_name,data.shape[0],data.shape[1]))
    print("--\tcleaning data -- ")
    data['cleaned_text'] = data['comment_text'].apply(text_clean)
    print(data.head())
    print("---\tfinished cleaning data (tokenize, lowercase, lemmatize, remove stop words) -- ")
