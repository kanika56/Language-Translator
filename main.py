import string
import re
import pandas as pd
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Embedding, Input, LSTM, Dense, RepeatVector
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from tensorflow.keras import optimizers 
import numpy as np

data_path = '/content/deu.txt'
with open(data_path, 'r', encoding = 'utf-8') as f:
  lines = f.read()

# split text in lines
def to_lines(text):
  line = text.strip().split("\n")
  line = [i.split("\t") for i in line]
  return line

ger_eng = to_lines(lines)
ger_eng = np.array(ger_eng)

ger_eng = ger_eng[:80000,:] # taking 80000 words from data
ger_eng = ger_eng[:,[0,1]] # removing 3rd column

# Data Cleaning

ger_eng[:,0] = [s.translate(str.maketrans('','', string.punctuation)) for s in ger_eng[:,0]]
ger_eng[:,1] = [s.translate(str.maketrans('','', string.punctuation)) for s in ger_eng[:,1]]

for i in range (len(ger_eng)):
  ger_eng[i, 0] = ger_eng[i, 0].lower()
  ger_eng[i, 1] = ger_eng[i, 1].lower()

# text to sequence

def tokenization(lines): # build tokenizer
  tokenizer = Tokenizer()
  tokenizer.fit_on_texts(lines)
  return tokenizer

eng_tokenizer = tokenization(ger_eng[:, 0])
eng_vocab_size = len(eng_tokenizer.word_index) + 1 # 7876
eng_length = 8

ger_tokenizer = tokenization(ger_eng[:, 1])
ger_vocab_size = len(ger_tokenizer.word_index) + 1 # 13609
ger_length = 8

# encode and pad sequences, padding to max sequence length

def encode_sequences(tokenizer, length, lines):
  # integer encode sequences
  seq = tokenizer.texts_to_sequences(lines)
  # pad sequences with zero values
  seq = pad_sequences(seq, maxlen= length, padding = 'post')
  return seq

from sklearn.model_selection import train_test_split
# split data into train and test
train, test = train_test_split(ger_eng, test_size = 0.2, random_state = 12)

# prepare trainig data
trainX = encode_sequences(ger_tokenizer, ger_length, train[:, 1])
trainY = encode_sequences(eng_tokenizer, eng_length, train[:, 0])

# prepare validation data
testX = encode_sequences(ger_tokenizer, ger_length, test[:, 1])
testY = encode_sequences(eng_tokenizer, eng_length, test[:, 0])

# build seq-to-seq architecture

def define_model(in_vocab, out_vocab, in_timesteps, out_timesteps, units):
  model = Sequential()
  model.add(Embedding(in_vocab, units, input_length = in_timesteps, mask_zero = True))
  model.add(LSTM(units))
  model.add(RepeatVector(out_timesteps))
  model.add(LSTM(units, return_sequences = True))
  model.add(Dense(out_vocab, activation = 'softmax'))
  return model

model = define_model(ger_vocab_size, eng_vocab_size, ger_length, eng_length, 512)
rms = keras.optimizers.RMSprop(lr = 0.001)
model.compile(optimizer = rms, loss = 'sparse_categorical_crossentropy')

# train the model
history = model.fit(trainX, trainY.reshape(trainY.shape[0], trainY.shape[1], 1), epochs = 50, batch_size = 512, validation_split = 0.2)

preds = np.argmax(model.predict(testX.reshape((testX.shape[0], testX.shape[1]))))

# integers to their corresponding words 
def get_word(n, tokenizer):
  for word, index in tokenizer.word_index.items():
    if index == n:
      return word
    return None

# predictions into sentences
preds_text = []
for i in preds:
  temp = []
  for j in range (len(i)):
    t = get_word(i[j], eng_tokenizer)
    if j > 0:
      if (t == get_word(i[j], eng_tokenizer) or t == None):
        temp.append('')
      else:
        temp.append(t)
    else:
      if (t == None):
        temp.append('')
      else:
        temp.append(t)
  preds_text.append(' '.join(temp))

pred_df = pd.DataFrame({'actual' : test[:, 0], 'predicted' : preds_text})
pred_df.sample(15)
