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
from sklearn.feature_extraction.text import TfidfVectorizer



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


ngram = 2
vectorizer = TfidfVectorizer(sublinear_tf=True,ngram_range=(1, ngram), max_df=0.5)


X_all = X_train + X_test # Combine both to fit the TFIDF vectorization.
lentrain = len(X_train)

vectorizer.fit(X_all) # This is the slow part!
X_all = vectorizer.transform(X_all)

X_train_chuli = X_all[:lentrain] # Separate back into training and test sets.
X_test_chuli = X_all[lentrain:]
X_train_chuli.shape


from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import SGDClassifier as SGD

folds = StratifiedKFold(n_splits=35, shuffle=False, random_state=2019)
oof = np.zeros(X_train_chuli.shape[0])
predictions = np.zeros(X_test_chuli.shape[0])

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train_chuli, y_train)):
    print("Fold :{}".format(fold_ + 1))
    trn_data = X_train_chuli[trn_idx]
    trn_label= y_train[trn_idx]
    val_data = X_train_chuli[val_idx]
    val_label= y_train[val_idx]
    model_SGD = SGD(alpha=0.00001,random_state = 2, shuffle = True, loss = 'log')
    model_SGD.fit(trn_data, trn_label) # Fit the model.
    print("auc score: {:<8.5f}".format(metrics.roc_auc_score(val_label, model_SGD.predict_proba(val_data)[:,1])))
    predictions += model_SGD.predict_proba(X_test_chuli)[:,1] / folds.n_splits

print(len(predictions))
predictions[:4]
SGD_output = pd.DataFrame({"ID":df_test["ID"], "Pred":predictions})
SGD_output.to_csv('SGD_new.csv', index = False)

