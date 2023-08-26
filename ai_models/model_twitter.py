API_Key = "66HFuF9CWNnCpsHHLtc5khnQ8"
API_Secret = "v7lBS0F03U8dTUKBuBq2urkiXadtND3cvzfQwgUu9xCvRMeclA"
bearer_token = "AAAAAAAAAAAAAAAAAAAAALpmawEAAAAAMVDl5ChgrVOTmOjeVU6idV%2BPeUY%3Di7bTnDc6SDnYCexe4jdOnLVUnMqdVaj6tKsAKb94k82kqVf6nQ"
Access_token = "1508300589192867845-7pxfiqYLvaqPbqQiOKSynFgsRp7Xo9"
Access_Secret = "S5btsUILPWRhzXMvd7GkCw3Bd5mHn0IgLJ9f6LpsdLhf0"
import tweepy
import os
q1 = '#tatamotors OR #TataMotors -is:Retweet -has:media lang:en'
q2 = '#tatapower OR #TataPower OR #Tatapower -is:Retweet -has:media lang:en'
q3 = '#tatachemicals OR #TataChemicals OR #Tatachemicals -is:Retweet -has:media lang:en'
q4 = '#BPCL -is:Retweet -has:media lang:en'
q5 = '#HindustanUnilever OR #hindustanunilever OR #HUL  -has:media lang:en '
import preprocessor as p
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re

translate_table = dict((ord(char), None) for char in string.punctuation)
lemmatizer = WordNetLemmatizer()


def extract_tweets(q):
    client = tweepy.Client(bearer_token=bearer_token)
    twt = client.search_recent_tweets(query=q, max_results=100)
    t = twt[0]
    for i in range(len(t)):
        t[i] = str(t[i])
    return t


def model(txt):
    finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone', num_labels=3)
    tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
    nlp = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer)
    result = nlp(txt)
    print(result)
    return result


def clean(twt):
    for i in range(len(twt)):
        twt[i] = twt[i].lower()
        twt[i] = twt[i].translate(translate_table)
        twt[i] = re.sub('[0-9]+', '', twt[i])

        return twt


def plot(r):
    c1 = 0
    c2 = 0
    c3 = 0
    for i in range(len(r)):
        if r[i]['label'] == 'Neutral':
            c2 += 1
        if r[i]['label'] == 'Positive':
            c1 += 1
        if r[i]['label'] == 'Negative':
            c3 += 1

    x = ['Positive', 'Negative', 'Neutral']
    y = [c1, c3, c2]

    df = pd.DataFrame({'sentiment': x, 'count': y})

    plott = sns.barplot(x='sentiment', y='count', data=df)
    plt.savefig('saved_figure.png')
    plt.show()

# select q1, q2, q3 .. based on the button pressed in the front end and pass that corresponding q to extract tweets function
# the output of extract tweets to be given to clean funciton and its output to be given to model
# model returns array of decisons which is given to plot which create a bar graph
def fn_main():
    plot(model(clean(extract_tweets(q2))))
    image = open('saved_figure.png', 'rb')
    return image

fn_main()