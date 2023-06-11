#!/usr/bin/env python
# coding: utf-8

# ### Pretrained Word Embedding
# 
# he Keras Embedding layer can also use a word embedding learned elsewhere.
# 
# It is common in the field of Natural Language Processing to learn, save, and make freely available word embeddings.
# 
# For example, the researchers behind GloVe method provide a suite of pre-trained word embeddings on their website released under a public domain license. See:
# 
# GloVe: Global Vectors for Word Representation
# The smallest package of embeddings is 822Mb, called “glove.6B.zip“. It was trained on a dataset of one billion tokens (words) with a vocabulary of 400 thousand words. There are a few different embedding vector sizes, including 50, 100, 200 and 300 dimensions.
# 
# You can download this collection of embeddings and we can seed the Keras Embedding layer with weights from the pre-trained embedding for the words in your training dataset.
# 
# After downloading and unzipping, you will see a few files, one of which is “glove.6B.100d.txt“, which contains a 100-dimensional version of the embedding.
# 
# If you peek inside the file, you will see a token (word) followed by the weights (100 numbers) on each line. For example, below are the first line of the embedding ASCII text file showing the embedding for “the“.

# In[2]:


from numpy import array
from numpy import asarray
from numpy import zeros
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Embedding


# In[3]:


# define documents
docs = ['Well done!',
        'Good work',
        'Great effort',
        'nice work',
        'Excellent!',
        'Weak',
        'Poor effort!',
        'not good',
        'poor work',
        'Could have done better.']
# define class labels
labels = array([1,1,1,1,1,0,0,0,0,0])


# ### Using Tokenizer Keras similar to OneHot Representation 

# In[4]:


# prepare tokenizer
t = Tokenizer()
t.fit_on_texts(docs)
vocab_size = len(t.word_index) + 1
# integer encode the documents
encoded_docs = t.texts_to_sequences(docs)
print(encoded_docs)


# ### Using Pad Sequences
# 

# In[5]:


# pad documents to a max length of 4 words
max_length = max([len(sen.split(' ')) for sen in docs ])
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
print(padded_docs)


# ### Load Glove Word Embedding File as Dictionary of Word to embedding array

# In[6]:


# load the whole embedding into memory
embeddings_index = dict()
f = open('glove.6B.100d.txt',encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))
print(type(embeddings_index))


# In[7]:


embeddings_index['well']  ### there will be 100 d of word 'well'


# In[8]:


vocab_size


# In[9]:


t.word_index.items() ### Each word has its own Integer value which is required before Embedding layer


# ### Creating Embedded Matrix with GLOVE weigths

# Next, we need to create a matrix of one embedding for each word in the training dataset. We can do that by enumerating all unique words in the Tokenizer.word_index and locating the embedding weight vector from the loaded GloVe embedding.

# In[10]:


embedding_matrix = zeros((vocab_size, 100))
print(embedding_matrix)
embedding_matrix.shape


# In[11]:


embedding_vector = embeddings_index.get('well')
embedding_vector


# Now above Embedded vector of word 'well' will get replace in Main Embedded Matrix 

# In[12]:


### Above process will be done for each and every word . Its value will get stored in Embedded_Matrix

for word, i in t.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

embedding_matrix


# ### Embedding Layer 
# 
# Now we will Directly provide the Embedded Matrix to the Embedding Layer which has weights from Glove
# 
# The key difference is that the embedding layer can be seeded with the GloVe word embedding weights. We chose the 100-dimensional version, therefore the Embedding layer must be defined with output_dim set to 100. Finally, we do not want to update the learned word weights in this model, therefore we will set the trainable attribute for the model to be False.
# 
# Here Learning will not be done , Becuase we have alreadY used pretrained glove embedding

# In[13]:


max_length


# In[14]:


e = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=max_length, trainable=False)


# In[15]:


model = Sequential()
model.add(e)
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# summarize the model
print(model.summary())


# In[ ]:


model.fit(padded_docs, labels, epochs=50, verbose=0)


# In[ ]:


# evaluate the model
loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
print('Accuracy: %f' % (accuracy*100))


# In[ ]:




