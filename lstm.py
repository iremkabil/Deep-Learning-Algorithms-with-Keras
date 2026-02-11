# %% load dataset and preprocessing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups
from sklearn.preprocessing import LabelEncoder # etiketleri sayisal formata cevirir
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import Tokenizer # metin verisini sayilara cevirir
from tensorflow.keras.preprocessing.sequence import pad_sequences # dizileri ayni uzunluga getirir
from tensorflow.keras.models import Sequential # kerasta model olusturma sinifi
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

import warnings
warnings.filterwarnings("ignore")

newsgroup = fetch_20newsgroups(subset = "all") # all ile hem egitim hem test verileri yuklenir
X = newsgroup.data
y = newsgroup.target

# Metin verisini tokenize etme ve padding

# Tüm metni tarıyor => Her kelimeyi sayiyor => En sık kelimelere index veriyor
# LSTM tum inputlarin ayni boyutlu olmasini ister.
# 100 kelimeden kisa ise basina 0 ekler. Fazla ise keser.

tokenizer = Tokenizer(num_words = 10000) # num_words = en cok kullanilan kelime sayisi
tokenizer.fit_on_texts(X) # tokenizer'i metin verisi ile fit edelim.
X_sequences =tokenizer.texts_to_sequences(X) # metinleri sayisala cevirir
X_padded = pad_sequences(X_sequences,maxlen = 100) # metinleri ayni uzunluga getirir


# Etiketleri sayisal hale donustur (label encoding)

# y icindeki tum unique siniflari bul => alfabetik siraya koy => sayiya cevir
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Tokenizer → X'i sayıya çevirir
# LabelEncoder → y'yi sayıya çevirir

# Train test split 

X_train, X_test, y_train, y_test = train_test_split(X_padded, y_encoded, test_size= 0.2, random_state=42)

# %% create (build) lstm model

def build_lstm_model():
    model = Sequential()
    
    # katmanlar: embedding + lstm + dropout + output
    
    # input_dim: kelime vektorlerin toplam boyutu
    # output_dim: kelime vektorlerin boyutu
    # input_length her giris metninin uzunlugu
    
    model.add(Embedding(input_dim =10000, output_dim=64, input_length = 100))
    
    # return_sequence: sonuclarin tum zaman adimlari yerine sadece son adimda return etmesi
    model.add(LSTM(units=64, return_sequences=False)) # 64 adet hucre, 
    
    model.add(Dropout(0.5))
    
    model.add(Dense(20, activation = "softmax")) # 20 sinif => 20 noron
    
    # model compile
    model.compile(optimizer="adam",
                   loss="sparse_categorical_crossentropy",
                   metrics=["accuracy"])
    return model
    
# model olusturma
model = build_lstm_model()
model.summary()

# %% train lstm

# callbacks = early stop

# model train

# %% model evaluation

# test veri seti ile degerlendirme

# history kullanarak accuracy ve loss degerlerini gorsellestirme
