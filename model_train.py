import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# set train dataset 
df = pd.read_csv("train.csv", encoding='latin1')

# I assigned the input and target value  (inout in  text and target label is sentiment)
texts = list(map(lambda x:str(x),df['text']))
sentiment = list(map(lambda x:str(x),df['sentiment']))
label_list = []
for i in sentiment:
  if i =="positive":
    label_list.append(1)
  elif i =="negative":
    label_list.append(0)
  else:
    label_list.append(2)
len(texts),len(label_list)

labels = np.array(label_list)


##  configuration for model training 
max_words = 10000 
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

max_sequence_length = 200 
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
embedding_dim = 100  
model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=max_sequence_length))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2)) 
model.add(Dense(64, activation='relu'))  
model.add(Dropout(0.5))  
model.add(Dense(3, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Now we fit the data for training

model.fit(padded_sequences, labels, epochs=10)

# Save the model
model.save("sentiment_model_New.h5")

# Save the tokenizer 
with open("tokenizer_new.pkl", "wb") as f:
    pickle.dump(tokenizer, f)