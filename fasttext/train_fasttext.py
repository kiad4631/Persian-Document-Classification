#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


data_xlsx = pd.read_excel("hamshahri_shuffled_notab.xlsx" , encoding = 'utf-8')


# In[2]:


data_xlsx.head()


# In[3]:


# for i in range(len(data_xlsx)):
#     if data_xlsx["CAT[2]/text()"][i].split('.')[0] == "Economy":
#         data_xlsx["CAT[2]/text()"][i] = "Economy"
#     if data_xlsx["CAT[2]/text()"][i].split('.')[0] == "Sport":
#         data_xlsx["CAT[2]/text()"][i] = "Sport"
#     if data_xlsx["CAT[2]/text()"][i].split('.')[0] == "Literature" or data_xlsx["CAT[2]/text()"][i].split(' ')[0] == "Literature":
#         data_xlsx["CAT[2]/text()"][i] = "Literature"
#     if data_xlsx["CAT[2]/text()"][i].split('.')[0] == "Miscellaneous":
#         data_xlsx["CAT[2]/text()"][i] = "Miscellaneous"
#     if data_xlsx["CAT[2]/text()"][i].split('.')[0] == "Politics":
#         data_xlsx["CAT[2]/text()"][i] = "Politics"
#     if data_xlsx["CAT[2]/text()"][i].split('.')[0] == "Social":
#         data_xlsx["CAT[2]/text()"][i] = "Social"


# In[4]:


for i in range(len(data_xlsx)):
    if data_xlsx["CAT[2]/text()"][i].split('.')[0] == "Economy":
        data_xlsx["CAT[2]/text()"][i] = "0"
    if data_xlsx["CAT[2]/text()"][i].split('.')[0] == "Sport":
        data_xlsx["CAT[2]/text()"][i] = "1"
    if data_xlsx["CAT[2]/text()"][i].split('.')[0] == "Literature" or data_xlsx["CAT[2]/text()"][i].split(' ')[0] == "Literature":
        data_xlsx["CAT[2]/text()"][i] = "2"
    if data_xlsx["CAT[2]/text()"][i].split('.')[0] == "Miscellaneous":
        data_xlsx["CAT[2]/text()"][i] = "3"
    if data_xlsx["CAT[2]/text()"][i].split('.')[0] == "Politics":
        data_xlsx["CAT[2]/text()"][i] = "4"
    if data_xlsx["CAT[2]/text()"][i].split('.')[0] == "Social":
        data_xlsx["CAT[2]/text()"][i] = "5"
    if data_xlsx["CAT[2]/text()"][i].split(' ')[0] == "Science":
        data_xlsx["CAT[2]/text()"][i] = "6"


# In[5]:


c= 0
for i in range(len(data_xlsx)):
    data_xlsx["TEXT[1]/text()"][i] = data_xlsx["TEXT[1]/text()"][i][:100]

        


# In[73]:


data_xlsx[:30]


# In[6]:


data_xlsx.head()


# In[7]:


f = open("embeding.txt", "r" , encoding="utf-8")
counter = 0
for line in f:
    if(counter == 10):
        break
    print(line)
    counter=counter+1
    


# In[8]:


import numpy as np
import matplotlib.pyplot as plt
import csv


# In[9]:


import keras


# In[10]:


def read_csv(filename):
    phrase = []
    emoji = []

    with open (filename , encoding="utf-8") as csvDataFile:
        csvReader = csv.reader(csvDataFile)

        for row in csvReader:
            phrase.append(row[0])
            emoji.append(row[1])

    X = np.asarray(phrase)
    Y = np.asarray(emoji)

    return X, Y


# In[11]:


data_xlsx.iloc[0:int(len(data_xlsx)*0.8)].to_csv('train.csv' , encoding = "utf-8" , index = False)
data_xlsx.iloc[int(len(data_xlsx)*0.8):].to_csv('test.csv' , encoding = "utf-8" , index = False)


# In[12]:


Y_train, X_train = read_csv('train.csv')
Y_test, X_test = read_csv('test.csv')


# In[13]:


Y_train = np.delete(Y_train,0)


# In[14]:


Y_train[:10]


# In[15]:


X_train = np.delete(X_train,0)


# In[16]:


X_train[:10]


# In[17]:


X_test = np.delete(X_test,0)


# In[18]:


Y_test = np.delete(Y_test,0)


# In[19]:


Y_test


# In[20]:


type(Y_train)


# In[21]:


for i in range(len(Y_test)):
    Y_test[i] = int(Y_test[i])


# In[22]:


for i in range(len(Y_train)):
    Y_train[i] = int(Y_train[i])


# In[23]:


Y_oh_train =  keras.utils.to_categorical(Y_train,  7)
Y_oh_test =  keras.utils.to_categorical(Y_test, 7)


# In[24]:



def read_fasttext_vecs(fasttext_file):
    with open(fasttext_file, encoding="utf-8") as f:
        c=0
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            #print(line[0])
            words.add(curr_word)
            c=c+1
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
        
        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in (words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map


# In[25]:


word_to_index, index_to_word, word_to_vec_map = read_fasttext_vecs('embeding.txt')


# In[36]:


word = "ورزش"
index = 12
print("the index of", word, "in the vocabulary is", word_to_index[word])
print("the", str(index) + "th word in the vocabulary is", index_to_word[index])


# In[87]:


def sentence_to_avg(sentence, word_to_vec_map):
    
    # Split sentence into list of lower case words
    words = sentence.split()

    # Initialize the average word vector, should have the same shape as your word vectors.
    avg = np.zeros((300,))
    
    # average the word vectors. You can loop over the words in the list "words".
    for w in words:
        
        try:
            vec=word_to_vec_map[w]
        except :
            vec=np.zeros((300,))
        avg += vec   
    avg = avg/len(words)
    
   
    return avg


# In[45]:


# def softmax(x):
#     """Compute softmax values for each sets of scores in x."""
#     e_x = np.exp(x - np.max(x))
#     return e_x / e_x.sum()

# def predict(X, Y, W, b, word_to_vec_map):
#     """
#     Given X (sentences) and Y (emoji indices), predict emojis and compute the accuracy of your model over the given set.
    
#     Arguments:
#     X -- input data containing sentences, numpy array of shape (m, None)
#     Y -- labels, containing index of the label emoji, numpy array of shape (m, 1)
    
#     Returns:
#     pred -- numpy array of shape (m, 1) with your predictions
#     """
#     m = X.shape[0]
#     pred = np.zeros((m, 1))
    
#     for j in range(m):                       # Loop over training examples
        
#         # Split jth test example (sentence) into list of lower case words
#         words = X[j].lower().split()
        
#         # Average words' vectors
#         avg = np.zeros((300,))
#         for w in words:
#             try:
#                 vec=word_to_vec_map[w]
#             except :
#                 vec=np.zeros((300,))
#             avg += vec   
#         avg = avg/len(words)

#         # Forward propagation
#         Z = np.dot(W, avg) + b
#         A = softmax(Z)
#         pred[j] = np.argmax(A)
        
#     print("Accuracy: "  + str(np.mean((pred[:] == Y.reshape(Y.shape[0],1)[:]))))
    
#     return pred

# def model(X, Y, word_to_vec_map, learning_rate = 0.01, num_iterations = 401):
#     """
#     Model to train word vector representations in numpy.
    
#     Arguments:
#     X -- input data, numpy array of sentences as strings, of shape (m, 1)
#     Y -- labels, numpy array of integers between 0 and 7, numpy-array of shape (m, 1)
#     word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
#     learning_rate -- learning_rate for the stochastic gradient descent algorithm
#     num_iterations -- number of iterations
    
#     Returns:
#     pred -- vector of predictions, numpy-array of shape (m, 1)
#     W -- weight matrix of the softmax layer, of shape (n_y, n_h)
#     b -- bias of the softmax layer, of shape (n_y,)
#     """
    
#     np.random.seed(1)

#     # Define number of training examples
#     m = Y.shape[0]                          # number of training examples
#     n_y = 7                                 # number of classes  
#     n_h = 300                               # dimensions of the GloVe vectors 
    
#     # Initialize parameters using Xavier initialization
#     W = np.random.randn(n_y, n_h) / np.sqrt(n_h)
#     b = np.zeros((n_y,))
    
#     # Convert Y to Y_onehot with n_y classes
#     Y_oh = keras.utils.to_categorical(Y, n_y) 
    
#     # Optimization loop
#     for t in range(num_iterations):                       # Loop over the number of iterations
#         for i in range(m):                                # Loop over the training examples
            
#             # Average the word vectors of the words from the i'th training example
#             avg = sentence_to_avg(X[i], word_to_vec_map)

#             # Forward propagate the avg through the softmax layer
#             z = np.dot(W, avg) + b
#             a = softmax(z)

#             # Compute cost using the i'th training label's one hot representation and "A" (the output of the softmax)
#             cost = -np.sum(Y_oh[i] * np.log(a))
            
#             # Compute gradients 
#             dz = a - Y_oh[i]
#             dW = np.dot(dz.reshape(n_y,1), avg.reshape(1, n_h))
#             db = dz

#             # Update parameters with Stochastic Gradient Descent
#             W = W - learning_rate * dW
#             b = b - learning_rate * db
        
#         if t % 100 == 0:
#             print("Epoch: " + str(t) + " --- cost = " + str(cost))
#             pred = predict(X, Y, W, b, word_to_vec_map)

#     return pred, W, b


# In[75]:


# avg = sentence_to_avg("ورزش خوب است", word_to_vec_map)
# print("avg = ", avg)


# In[76]:


# pred, W, b = model(X_train, Y_train, word_to_vec_map)


# In[70]:


avg_train=[]
for i in range(len(X_train)):
    avg_train.append(sentence_to_avg(X_train[i],word_to_vec_map))


# In[65]:


avg_test=[]
for i in range(len(X_test)):
    avg_test.append(sentence_to_avg(X_test[i],word_to_vec_map))


# In[66]:


from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop


# In[67]:


batch_size = 128
num_classes = 7
epochs = 20


# In[71]:


len(avg_train)


# In[120]:


np.array(avg_train)


# In[74]:


model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(300,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(7, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(np.array(avg_train), Y_oh_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1)
score = model.evaluate(np.array(avg_test), Y_oh_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[156]:


t=X_test[2621]

pred = model.predict(np.expand_dims(sentence_to_avg(t,word_to_vec_map), axis = 0))


# In[157]:


np.argmax(np.squeeze(pred))


# In[158]:


Y_test[2621]


# In[ ]:




