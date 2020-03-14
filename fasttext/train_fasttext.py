import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

#Read excel data
data_xlsx = pd.read_excel("full_hamshahri.xlsx" , encoding = 'utf-8')
data_xlsx.head()

#Convert labels of documents to numeric values
for i in range(len(data_xlsx)):
    if data_xlsx["CAT[2]/text()"][i].split('.')[0] == "Economy":
        data_xlsx["CAT[2]/text()"][i] = "0"
    if data_xlsx["CAT[2]/text()"][i].split('.')[0] == "Sport":
        data_xlsx["CAT[2]/text()"][i] = "1"
    if data_xlsx["CAT[2]/text()"][i].split('.')[0] == "Literature" or data_xlsx["CAT[2]/text()"][i].split(' ')[0] == "Literature":
        data_xlsx["CAT[2]/text()"][i] = "2"
    if data_xlsx["CAT[2]/text()"][i].split('.')[0] == "Politics":
        data_xlsx["CAT[2]/text()"][i] = "3"
    if data_xlsx["CAT[2]/text()"][i].split('.')[0] == "Social":
        data_xlsx["CAT[2]/text()"][i] = "4"
    if data_xlsx["CAT[2]/text()"][i].split(' ')[0] == "Science":
        data_xlsx["CAT[2]/text()"][i] = "5"


#Take 100 first charachters of each document 
for i in range(len(data_xlsx)):
    data_xlsx["TEXT[1]/text()"][i] = data_xlsx["TEXT[1]/text()"][i][:100]

#Read train and test csv files and return data and corresponding labels
def read_csv(filename):
    text = []
    label = []
    with open (filename , encoding="utf-8") as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            text.append(row[0])
            label.append(row[1])
    X = np.asarray(text)
    Y = np.asarray(label)
    return X, Y

#Split train and test data with ratio 0.8, 0.2 respectively
data_xlsx.iloc[0:int(len(data_xlsx)*0.8)].to_csv('train.csv' , encoding = "utf-8" , index = False)
data_xlsx.iloc[int(len(data_xlsx)*0.8):].to_csv('test.csv' , encoding = "utf-8" , index = False)

#Set data and label for both train and test data
Y_train, X_train = read_csv('train.csv')
Y_test, X_test = read_csv('test.csv')

#Delete headers of train and test sets---The header of X_train and X_test is : TEXT[1]/text() and the header of Y_train and Y_test is : CAT[2]/text()
Y_train = np.delete(Y_train,0)
X_train = np.delete(X_train,0)
X_test = np.delete(X_test,0)
Y_test = np.delete(Y_test,0)

#Convert labels to integer
for i in range(len(Y_test)):
    Y_test[i] = int(Y_test[i])
for i in range(len(Y_train)):
    Y_train[i] = int(Y_train[i])

#Prepare lables for training in Keras
Y_oh_train =  keras.utils.to_categorical(Y_train,  7)
Y_oh_test =  keras.utils.to_categorical(Y_test, 7)

#Helper function for pre-trained embedding reading
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

#Dictionary mapping words to their fasstest vector representation
word_to_index, index_to_word, word_to_vec_map = read_fasttext_vecs('embeding.txt')

#Helper function for converting a sentence to the average embedding of words in the sentence
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

#Average embedding of words in each row of train and test data as input of the model
avg_train=[]
for i in range(len(X_train)):
    avg_train.append(sentence_to_avg(X_train[i],word_to_vec_map))
avg_test=[]
for i in range(len(X_test)):
    avg_test.append(sentence_to_avg(X_test[i],word_to_vec_map))

#Model architecture and train
batch_size = 128
num_classes = 7
epochs = 20
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
model.save('my_model.h5')
#Print evaluation model on test data
score = model.evaluate(np.array(avg_test), Y_oh_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
