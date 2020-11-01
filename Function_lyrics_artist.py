#!/usr/bin/env python
# coding: utf-8

# In[10]:


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
import random
from tqdm import tqdm_notebook

from bs4 import BeautifulSoup

from IPython.display import HTML, display
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
from datetime import datetime

from sklearn.model_selection import KFold
import spacy
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from Function_list_artist_link_lyrics import list_artist_link_lyrics
from Function_link_artist_lyrics import link_lyrics_artist


# In[11]:


def lyrics_artist(artist):
    '''
    Create a DataFrame of lyrics of artist songs.
    '''
    lyrics_artist = []
    df = link_lyrics_artist(artist)
    for link in tqdm_notebook(df['song_link']):
        link_url = f" https://www.lyrics.com{link}"
        resp_lyrics = requests.get(link_url)
        time.sleep(random.random()/10+.01)
        soup_lyrics = BeautifulSoup(resp_lyrics.text)
        lyrics_text = soup_lyrics.find('pre', id ='lyric-body-text').get_text()
        #print(lyrics_text, song_name)
        lyrics_artist.append(lyrics_text)
        
        # Add the artist name and remove duplications with lyrics
        df['lyrics']= lyrics_artist
        df['artist_name']= 'artist'
        df = df.drop_duplicates(subset=['lyrics'])

    
    return df 
    
   


# In[12]:


lyrics_artist('Mehul-Kumar')

