####
from data_process import read_file
#####
import re
import numpy as np
import pandas as pd
import bilstm
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

word_size = 128
maxlen = 375
filename='D:/pythonProject/test/NLP/data/training.txt'
_, content, label1 = read_file(filename)

data = content
label = label1


d = pd.DataFrame(index=range(len(data)))
d['data'] = data
d['label'] = label
d = d[d['data'].apply(len) <= maxlen]
d.index = range(len(d))
tag = pd.Series({'S': 0, 'B': 1, 'M': 2, 'E': 3, 'X': 4})

dic= []
for i in data:
    dic.extend(i)

dic = pd.Series(dic).value_counts()
dic[:] = range(1, len(dic) + 1)

import pickle

with open('model/dic.pkl', 'wb') as outp:
    pickle.dump(dic, outp)

from keras.utils import np_utils

d['x'] = d['data'].apply(lambda x: np.array(list(dic[x]) + [0] * (maxlen - len(x))))

def trans_one(x):
    _ = map(lambda y: np_utils.to_categorical(y, 5), tag[x].values.reshape((-1, 1)))
    _ = list(_)
    _.extend([np.array([[0, 0, 0, 0, 1]])] * (maxlen - len(x)))
    return np.array(_)


d['y'] = d['label'].apply(trans_one)


def train_bilstm():
    model = lstm_model.create_model(maxlen, dic, word_size)
    batch_size = 1024
    history = model.fit(np.array(list(d['x'])), np.array(list(d['y'])).reshape((-1, maxlen, 5)), batch_size=batch_size,
                        epochs=100, verbose=2)
    model.save('model/model.h5')

train_bilstm()




