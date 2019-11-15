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
    lemmatize = lambda token_list : [ lemmatizer.lemmatize(token) for token in token_list \
                                    if token not in stopwords.words('english') and len(token) > 3] 
    # def lemmatize(token_list):
    #     lemmatize =
    #     for token in token_list:
    #         token = lemmatizer.lemmatize(token)
    #         if token not in stopwords.words('english') and len(token) > 3:
    #             lemmatize.append(token)
    #     return lemmatize
    text = text.lower()
    token_words=word_tokenize(text)
    return lemmatize(token_words)

    # cleaned_tweet 	= []
    # tweet			= tweet_obj._json
    # raw_text		= emoji.demojize(tweet['full_text'])
    # cleaned_text 	= p.clean(raw_text)
    # cleaned_text    = lemmatize(cleaned_text)
    # cleaned_tweet 	+= [tweet['id'],'tweet', tweet['created_at'],tweet['source'], tweet['full_text'],cleaned_text,tweet['favorite_count'], tweet['retweet_count']]
    # hashtags = ", ".join([hashtag_item['text'] for hashtag_item in tweet['entities']['hashtags']])
    # cleaned_tweet.append(hashtags) #append hashtags 
    # mentions = ", ".join([mention['screen_name'] for mention in tweet['entities']['user_mentions']])
    # cleaned_tweet.append(mentions) #append mentions
    # cleaned_tweet.append(tweet['user']['screen_name'])
    # single_tweet_df = pd.DataFrame([cleaned_tweet], columns=COLS)
    #return single_tweet_df


if __name__ == "__main__":
    file_name = "train" if len(sys.argv) <=1 else sys.argv[1] #User didn't enter filename from command line
       
    # IMPORT DATA
    data = pd.read_csv("./data/{}.csv".format(file_name)).head(10)
    print("---\tread in {}.csv, Size: {}, number of attributes: {} ---".format(file_name,data.shape[0],data.shape[1]))
    print("--\tcleaning data -- ")
    data['cleaned_text'] = data['comment_text'].apply(text_clean)
    print(data.head())
    print("---\tfinished cleaning data (tokenize, lowercase, lemmatize, remove stop words) -- ")
