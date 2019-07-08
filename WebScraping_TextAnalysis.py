# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 00:17:35 2019

@author: jamielu
"""
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

        
# count Word Frequency, nltk offers a function FreqDist()        
freq = nltk.FreqDist(lemm_tokens)
for key,val in freq.items():
    print(str(key) + ':' + str(val))
freq.plot(30, cumulative=False)

# create a wordcloud
from wordcloud import WordCloud
import matplotlib.pyplot as plt
wc = WordCloud().generate_from_frequencies(freq)
plt.imshow(wc)
plt.axis("off")
plt.show()