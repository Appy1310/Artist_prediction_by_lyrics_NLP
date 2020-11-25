from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from imblearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import make_column_transformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pickle
import json
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import re
import spacy
from sklearn.model_selection import KFold
from datetime import datetime
import requests
import time

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import requests
import time


from IPython.display import HTML, display
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300


#from sklearn.pipeline import make_pipeline


def add_spacy_data(dataset, feature_column):
    '''
    Grabs the verb, adverb, noun, and stop word Parts of Speech (POS)
    tokens and pushes them into a new dataset. returns an
    enriched dataset'''
    verbs = []
    nouns = []
    adverbs = []
    corpus = []
    nlp = spacy.load('en_core_web_md')
    ##
    for i in range(0, len(dataset)):
        #print("Extracting verbs and topics from record {} of {}".format(i+1, len(dataset)), end = "\r")
        song = dataset.iloc[i][feature_column]
        doc = nlp(song)
        spacy_dataframe = pd.DataFrame()
        for token in doc:
            if token.lemma_ == "-PRON-":
                lemma = token.text
            else:
                lemma = token.lemma_
            row = {
                "Word": token.text,
                "Lemma": lemma,
                "PoS": token.pos_,
                "Stop Word": token.is_stop
            }
            spacy_dataframe = spacy_dataframe.append(row, ignore_index=True)
        verbs.append(
            " ".join(spacy_dataframe["Lemma"][spacy_dataframe["PoS"] == "VERB"].values))
        nouns.append(
            " ".join(spacy_dataframe["Lemma"][spacy_dataframe["PoS"] == "NOUN"].values))
        adverbs.append(
            " ".join(spacy_dataframe["Lemma"][spacy_dataframe["PoS"] == "ADV"].values))
        corpus_clean = " ".join(
            spacy_dataframe["Lemma"][spacy_dataframe["Stop Word"] == False].values)
        corpus_clean = re.sub(r'[^A-Za-z0-9]+', ' ', corpus_clean)
        corpus.append(corpus_clean)
    dataset['Verbs'] = verbs
    dataset['Nouns'] = nouns
    dataset['Adverbs'] = adverbs
    dataset['Corpus'] = corpus
    return dataset


with open('cv_rf', 'rb') as f:
    cv_rf = pickle.load(f)

# Take the lyrics as input
sample = input('Give me some lyrics: ')

X_sample = pd.DataFrame({'lyrics': [f'{sample}']}).reset_index()


prepped_test_data = add_spacy_data(X_sample, 'lyrics')
y_sample = cv_rf.best_estimator_.predict_proba(prepped_test_data)

print(y_sample)
print(cv_rf.classes_)
