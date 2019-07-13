# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jul  13 15:17:35 2019

@author: jamielu
"""
#https://medium.com/@nicharuch/collocations-identifying-phrases-that-act-like-individual-words-in-nlp-f58a93a2f84a

# Download Python Natural language toolkit (NLTK) library
import nltk
nltk.download()

#This will show the NLTK downloader to choose what packages need to be installed.

#urllib module will help us to crawl the webpage
import urllib.request
response =  urllib.request.urlopen('https://www.sparknotes.com/lit/littleprince/section7/')
html = response.read()
print(html)

# We will use Beautiful Soup which is a Python library for pulling data out of HTML and XML files. 
# BeautifulSoup provides a simple way to find text content (i.e. non-HTML) from the HTML:
from bs4 import BeautifulSoup
soup = BeautifulSoup(html,'html.parser')
text = soup.find_all(text = True)
print(text)

# There are a few items in here that we likely do not want:
output = ''
blacklist = [
	'[document]',
	'noscript',
	'header',
	'html',
	'meta',
	'head', 
	'input',
	'script',
    'style',
	# there may be more elements you don't want, such as "style", etc.
]

for t in text:
	if t.parent.name not in blacklist:
		output += '{} '.format(t)

print(output)

# convert output text to lowercase
output_lower = output.lower()


# Now we have a list of lowercase text crawled from the web page, let’s convert the output_lower into word tokens
from nltk.tokenize import word_tokenize 
word_tokens = word_tokenize(output_lower)

#from nltk.tokenize import sent_tokenize 
#sent_tokens = sent_tokenize(output_lower)

# remove punctuation from word_tokens
from string import punctuation
word_tokens = [''.join(c for c in s if c not in punctuation) for s in word_tokens]
# remove empty strings
word_tokens = [s for s in word_tokens if s]
 
# Removing stop words — frequent words such as ”the”, ”is”, etc. that do not have specific semantic to further cleanup the text corpus.
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
filtered_tokens = [w for w in word_tokens if not w in stop_words]

# Lemmatisation unlike Stemming, reduces the inflected words properly ensuring that the root word belongs to the language.
from nltk.stem import WordNetLemmatizer
# init the wordnet lemmatizer
lmtzr = WordNetLemmatizer()
lemm_tokens = [lmtzr.lemmatize(x) for x in filtered_tokens]


import nltk
bigrams = nltk.collocations.BigramAssocMeasures()
trigrams = nltk.collocations.TrigramAssocMeasures()
bigramFinder = nltk.collocations.BigramCollocationFinder.from_words(word_tokens)
trigramFinder = nltk.collocations.TrigramCollocationFinder.from_words(word_tokens)

import pandas as pd

#bigrams
bigram_freq = bigramFinder.ngram_fd.items()
bigramFreqTable = pd.DataFrame(list(bigram_freq), columns=['bigram','freq']).sort_values(by='freq', ascending=False)
#trigrams
trigram_freq = trigramFinder.ngram_fd.items()
trigramFreqTable = pd.DataFrame(list(trigram_freq), columns=['trigram','freq']).sort_values(by='freq', ascending=False)

#get english stopwords
en_stopwords = set(stopwords.words('english'))
#function to filter for ADJ/NN bigrams
def rightTypes(ngram):
    if '-pron-' in ngram or 't' in ngram:
        return False
    for word in ngram:
        if word in en_stopwords or word.isspace():
            return False
    acceptable_types = ('JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS')
    second_type = ('NN', 'NNS', 'NNP', 'NNPS')
    tags = nltk.pos_tag(ngram)
    if tags[0][1] in acceptable_types and tags[1][1] in second_type:
        return True
    else:
        return False
#filter bigrams
filtered_bi = bigramFreqTable[bigramFreqTable.bigram.map(lambda x: rightTypes(x))]

#function to filter for trigrams
def rightTypesTri(ngram):
    if '-pron-' in ngram or 't' in ngram:
        return False
    for word in ngram:
        if word in en_stopwords or word.isspace():
            return False
    first_type = ('JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS')
    third_type = ('JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS')
    tags = nltk.pos_tag(ngram)
    if tags[0][1] in first_type and tags[2][1] in third_type:
        return True
    else:
        return False
    
#filter trigrams
filtered_tri = trigramFreqTable[trigramFreqTable.trigram.map(lambda x: rightTypesTri(x))]



listbg=list(bigram_freq)


#https://pythonprogramming.net/part-of-speech-tagging-nltk-tutorial/
















