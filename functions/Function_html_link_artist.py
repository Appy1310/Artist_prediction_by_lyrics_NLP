#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


from IPython.display import HTML, display
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300


# ## Function for fetching html from Web

# In[3]:


def html_link_artist(artist):
    '''
    Fetches html link of the artist and saves the html in the folder.
    The input has to be given as a string.

    '''
    # Contrsuction of the whole url

    url_artist = f"https://www.lyrics.com/artist/{artist}"
    resp_artist = requests.get(url_artist)
    time.sleep(random.random() / 10 + .01)

    # Write as a html file in the local folder
    with open(f'./{artist}.html', 'w') as file:
        file.write(resp_artist.text)
