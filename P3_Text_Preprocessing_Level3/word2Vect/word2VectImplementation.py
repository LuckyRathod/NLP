# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 18:43:01 2020

@author: Lucky_Rathod
"""

####### WORD2VECT Implementation using GENSIM Library

import nltk

from gensim.models import Word2Vec
from nltk.corpus import stopwords

import re 

paragraph = """I have three visions for India. In 3000 years of our history, people from all over 
               the world have come and invaded us, captured our lands, conquered our minds. 
               From Alexander onwards, the Greeks, the Turks, the Moguls, the Portuguese, the British,
               the French, the Dutch, all of them came and looted us, took over what was ours. 
               Yet we have not done this to any other nation. We have not conquered anyone. 
               We have not grabbed their land, their culture, 
               their history and tried to enforce our way of life on them. 
               Why? Because we respect the freedom of others.That is why my 
               first vision is that of freedom. I believe that India got its first vision of 
               this in 1857, when we started the War of Independence. It is this freedom that
               we must protect and nurture and build on. If we are not free, no one will respect us.
               My second vision for India’s development. For fifty years we have been a developing nation.
               It is time we see ourselves as a developed nation. We are among the top 5 nations of the world
               in terms of GDP. We have a 10 percent growth rate in most areas. Our poverty levels are falling.
               Our achievements are being globally recognised today. Yet we lack the self-confidence to
               see ourselves as a developed nation, self-reliant and self-assured. Isn’t this incorrect?
               I have a third vision. India must stand up to the world. Because I believe that unless India 
               stands up to the world, no one will respect us. Only strength respects strength. We must be 
               strong not only as a military power but also as an economic power. Both must go hand-in-hand. 
               My good fortune was to have worked with three great minds. Dr. Vikram Sarabhai of the Dept. of 
               space, Professor Satish Dhawan, who succeeded him and Dr. Brahm Prakash, father of nuclear material.
               I was lucky to have worked with all three of them closely and consider this the great opportunity of my life. 
               I see four milestones in my career """
               
### Text Preprocessing 
text = re.sub(r'\[[0-9]*\]',' ',paragraph)
text = re.sub(r'\s+',' ',text)
text = text.lower()
text = re.sub(r'\d',' ',text)
text = re.sub(r'\s+',' ',text)

from nltk.stem.porter  import PorterStemmer
from nltk.stem import WordNetLemmatizer

ps = PorterStemmer()
wordnet = WordNetLemmatizer()
sentences = nltk.sent_tokenize(text)
corpus=[]

### Word2Vect Accepts list of sentences which again contains list of words i-e List of List

for i in range(len(sentences)):
    review = re.sub('[^a-zA-Z]',' ',sentences[i]) #Removing Punctuation -- Replace value other than expression with space
    review = review.split()
    ### Stemming
    #review_stem = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    ### Lemmatization
    review = [wordnet.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    #review = ' '.join(review)
    corpus.append(review)

### Training Word2Vect Model
'''
The model is fit when constructing the class. We pass in the list of clean sentences
from the training data, then specify the size of the embedding vector space (we use 100 again),
the number of neighboring words to look at when learning how to embed each word in the training
sentences (we use 5 neighbors), the number of threads to use when fitting the model (we use 8, but change 
this if you have more or less CPU cores), and the minimum occurrence count for words to consider in the vocabulary
(we set this to 1 as we have already prepared the vocabulary).
'''

# train word2vec model
model = Word2Vec(corpus, size=100, window=5, workers=8, min_count=1)#If word is present less than 1 times remove it

### Each (key) word in the dictionary will have a vector of 100 Dimension
words = model.wv.vocab 

### Finding the Word Vector
war_vector = model.wv['war']  ## You will get Vector of 100 Dimension of word WAR

### Most Similar word
similar = model.wv.most_similar('war')












