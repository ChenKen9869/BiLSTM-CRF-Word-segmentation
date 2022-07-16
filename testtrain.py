from data_process import read_file
import re
import numpy as np
import pandas as pd
import bilstm
import tensorflow as tf
from keras.models import load_model
import os

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

from val import simple_cut
from val import cutWord
save_path = r'D:\pythonProject\test\NLP\model\model.h5'
model = load_model(save_path)


file = open('D:/pythonProject/test/NLP/data/predict.txt', 'w',encoding='utf-8')

with open('D:/pythonProject/test/NLP/data/test.txt','r',encoding='utf-8') as thefile:
  for line in thefile:
      con = list(line)
      while ' ' in con:
          con.remove(' ')
      if '\n' in con:
          con.remove('\n')
      constr = ''.join(con)
      predict = cutWord(constr)
      predict.append('\n')
      #print(predict)
      for i in range(len(predict)):
         s = str(predict[i]) + '  '
         file.write(s)

file.close()


fInputHandle = open ('D:/pythonProject/test/NLP/data/predict.txt','r',encoding='utf-8')
fOutputHandle = open ('D:/NLP课程/第一次作业(1)/第一次作业/scripts/test_predict.txt', 'w+',encoding='utf-8')
lines = fInputHandle.readlines()
for line in lines:
    fOutputHandle.write(line.lstrip())

fInputHandle.close()
fOutputHandle.close()







