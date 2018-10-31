import numpy as np
import pandas as pd
import re
import csv

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import *
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import jaccard_similarity_score

labels = ['trust', 'surprise', 'sadness', 'pessimism', 'optimism', 'love',
          'joy', 'fear', 'disgust', 'anticipation', 'anger']
header = "ID\tTweet\tanger\tanticipation\tdisgust\tfear\tjoy\tlove\toptimism\tpessimism\tsadness\tsurprise\ttrust\n"

batch_size = 300
max_features=2000
max_len=200

def make_dictionary(filename):
    devset = {}
    index=0
    with open(filename) as f:
        for line in f:
            line = line.split("\t")
            devset[index] = [line[0], line[1]]
            index += 1
    return devset

def file_writer(filename):
    # starts at index 1 to ignore the header line.
    index = 1
    with open(filename, 'w', encoding='utf8') as f:
        f.write(header)
        # for all data in the devset, write line by line to file
        for data in devset:
            # write the ID and the Tweet
            f.write(devset[index][0] + "\t" + devset[index][1] + "\t")
            # write the zeros
            for i in range(10):
                f.write(str(int(round(0))) + "\t")
            f.write(str(int(round(0))) + "\n")
            index += 1

eps=20
def get_nn():
    embed_dim=64
    model = Sequential()
    model.add(Embedding(max_features, embed_dim, input_length=200))
    model.add(LSTM(100, activation='tanh'))
    model.add(Dense(11, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
        #model.compile(loss='binary_crossentropy',
        #          optimizer='adam',
        #          metrics=['accuracy'])
    return model

filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~'

# Train the data on data/train/txt
data = pd.read_csv("data/train.txt", sep="\t")
tokenizer = Tokenizer(num_words=max_features, split=' ', filters=filters)
tokenizer.fit_on_texts(data['Tweet'].values)
X_train = tokenizer.texts_to_sequences(data['Tweet'].values)
X_train = pad_sequences(X_train, maxlen=max_len)
Y_train = data.loc[:, labels].values

data = pd.read_csv("data/dev.txt", sep="\t")
X_dev = tokenizer.texts_to_sequences(data['Tweet'].values)
X_dev = pad_sequences(X_dev, maxlen=max_len)
Y_dev = data.loc[:, labels].values

model = get_nn()
model.fit(X_train, Y_train, epochs=eps, batch_size=batch_size, verbose=2, validation_data=(X_dev, Y_dev))

prediction = model.predict(X_dev)
prediction = (prediction > 0.5).astype(int)
print("Accuracy: {:.4f}%".format(100*jaccard_similarity_score(Y_dev, prediction)))
devset = make_dictionary("data/dev.txt")
index = 1
with open('output.txt', 'w', encoding='utf8') as f:
    f.write(header)
    for data in prediction:
        f.write(devset[index][0] + "\t" + devset[index][1] + "\t")
        for i in range(10):
            f.write(str(int(round(data[i]))) + "\t")
        f.write(str(int(round(data[-1]))) + "\n")
        index += 1
