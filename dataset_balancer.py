import numpy as np
import pandas as pd
from sklearn.utils import resample, shuffle



'''
This module takes in the cleaned dataset and balances it. Steps taken:

    1. Separate toxic comments from non toxic ones
    2. Undersample non-toxic comments by half
    3. Oversample toxic comments with replacement to match the number of non toxic comments
    4. Now the dataset consists of: |non-toxic comments| = 71673 |non-toxic comments| = 71673 
'''
if __name__ == "__main__":
    print('Worked')
    train_data = pd.read_csv('./data/cleaned_train.csv')
    print("Length of entire dataset: ", len(train_data))
    toxic_comments = train_data[(train_data['toxic'] != 0) | (train_data['toxic'] != 0) | (train_data['obscene'] != 0) | (train_data['threat'] != 0) | (train_data['insult'] != 0) | (train_data['identity_hate'] != 0)   ]
    non_toxic_comments = train_data[(train_data['toxic']== 0) & (train_data['toxic']== 0) & (train_data['obscene']== 0) & (train_data['threat']== 0) & (train_data['insult']== 0) & (train_data['identity_hate'] == 0)   ]

    non_toxic_comments_downsampled = non_toxic_comments.sample(n=71673)

    toxic_comments_upsampled = resample(toxic_comments, 
                                 replace=True,     # sample with replacement
                                 n_samples=71673)
    df_upsampled_downsampled = pd.concat([non_toxic_comments_downsampled,toxic_comments_upsampled])
    df_upsampled_downsampled = shuffle(df_upsampled_downsampled)
    df_upsampled_downsampled.to_csv('./data/upsampled_downsampled_train.csv')
    