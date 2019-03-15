import pandas as pd
import warnings;warnings.filterwarnings('ignore')
import re
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
# from keras.models import Sequential
# from keras.layers import Dense, Embedding, LSTM,Flatten
# from sklearn.model_selection import train_test_split
# from keras.utils.np_utils import to_categorical
import tensorflow as tf
from tensorflow import keras

import numpy as np




df_train = pd.read_csv('train.csv', lineterminator='\n')
df_test = pd.read_csv('test.csv', lineterminator='\n')

df_train['label'] = df_train['label'].map({'Negative': 0, 'Positive': 1})

# print(df_train.head())

print(df_train.isnull().sum())

# array_cleaner
numpy_array = df_train.as_matrix()
numpy_array_test = df_test.as_matrix()


#two commom ways to clean data
def cleaner(word):
  word = re.sub(r'\#\.', '', word)
  word = re.sub(r'\n', '', word)
  word = re.sub(r',', '', word)
  word = re.sub(r'\-', ' ', word)
  word = re.sub(r'\.', '', word)
  word = re.sub(r'\\', ' ', word)
  word = re.sub(r'\\x\.+', '', word)
  word = re.sub(r'\d', '', word)
  word = re.sub(r'^_.', '', word)
  word = re.sub(r'_', ' ', word)
  word = re.sub(r'^ ', '', word)
  word = re.sub(r' $', '', word)
  word = re.sub(r'\?', '', word)

  return word.lower()
def hashing(word):
  word = re.sub(r'ain$', r'ein', word)
  word = re.sub(r'ai', r'ae', word)
  word = re.sub(r'ay$', r'e', word)
  word = re.sub(r'ey$', r'e', word)
  word = re.sub(r'ie$', r'y', word)
  word = re.sub(r'^es', r'is', word)
  word = re.sub(r'a+', r'a', word)
  word = re.sub(r'j+', r'j', word)
  word = re.sub(r'd+', r'd', word)
  word = re.sub(r'u', r'o', word)
  word = re.sub(r'o+', r'o', word)
  word = re.sub(r'ee+', r'i', word)
  if not re.match(r'ar', word):
    word = re.sub(r'ar', r'r', word)
  word = re.sub(r'iy+', r'i', word)
  word = re.sub(r'ih+', r'eh', word)
  word = re.sub(r's+', r's', word)
  if re.search(r'[rst]y', 'word') and word[-1] != 'y':
    word = re.sub(r'y', r'i', word)
  if re.search(r'[bcdefghijklmnopqrtuvwxyz]i', word):
    word = re.sub(r'i$', r'y', word)
  if re.search(r'[acefghijlmnoqrstuvwxyz]h', word):
    word = re.sub(r'h', '', word)
  word = re.sub(r'k', r'q', word)
  return word


def array_cleaner(array):
  # X = array
  X = []
  for sentence in array:
    clean_sentence = ''
    words = sentence.split(' ')
    for word in words:
      clean_sentence = clean_sentence +' '+ cleaner(word)
    X.append(clean_sentence)
  return X


X_test = numpy_array_test[:, 1]
X_train = numpy_array[:, 1]
# Clean X here
X_train = array_cleaner(X_train)
X_test = array_cleaner(X_test)
y_train = numpy_array[:, 2]

y_train = np.array(y_train)
y_train = y_train.astype('int8')

X_all = X_train + X_test # Combine both to fit the tokenizer.
lentrain = len(X_train)


tokenizer = Tokenizer(
    nb_words=2000,
                      filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                   lower=True,split=' ')
tokenizer.fit_on_texts(X_all)


X = tokenizer.texts_to_sequences(X_all)
X = pad_sequences(X)


X_train = X[:lentrain] # Separate back into training and test sets.
X_test = X[lentrain:]


vocab_size = 2000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.summary()

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train,
                    y_train,
                    epochs=100,
                    batch_size=512,
                    verbose=1)

y_test = model.predict(X_test)[:, 0]

print(len(y_test))


lstm_output = pd.DataFrame(data={"ID":df_test["ID"], "Pred":y_test})
lstm_output.to_csv('lstm_new.csv', index = False, quoting = 3)

