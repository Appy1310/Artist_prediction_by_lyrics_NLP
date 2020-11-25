#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pickle
import json
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import re
import spacy
from sklearn.model_selection import KFold
from datetime import datetime
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

from bs4 import BeautifulSoup

from IPython.display import HTML, display
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300


# In[31]:


def list_artist_link_lyrics(artist):
    '''
    Fetches list of links of songs of the artist in www.lyrics.com and returns a list containing
    link of the lyrics and song names. The input has to be given as a string.

    '''
    # Reading the html file
    HtmlFile_artist = open(f"{artist}.html", 'r', encoding='utf-8')
    source_code_artist = HtmlFile_artist.read()

    # Use BeautySoup to find the lyrics links
    # Create a BeautifulSoup object

    soup_artist = BeautifulSoup(source_code_artist)
    song_links_artist = soup_artist.find_all('td', class_="tal qx")
    return song_links_artist


# In[32]:
list_artist_link_lyrics('Mehul-Kumar')


# In[33]:


def link_lyrics_artist(artist):
    '''
    Create a DataFrame of links of artist song lyrics.

    '''

    links_artist = []
    for link_tag in list_artist_link_lyrics(artist):
        link = link_tag.find('a').get('href')
        link_text = link_tag.get_text()
       # print(link, link_text)
        links_artist.append((link, link_text))

    df_artist = pd.DataFrame(links_artist, columns=['song_link', 'song'])
    df_artist.drop_duplicates(inplace=True, subset=['song'], ignore_index=True)
    return df_artist


# In[34]:


link_lyrics_artist('Mehul-Kumar')
