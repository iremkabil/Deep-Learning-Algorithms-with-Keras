# %% load data and preprocessing
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# veri seti yukleme
max_features = 10000 # en cok kullanilan 10000 kelimeyi kullan
maxlen = 100 # her yorumun max uzunlugu

# imdb
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = max_features)

# yorumlarin uzunluklarini max 100 olacak sekilde ayarla
x_train = pad_sequences(x_train, maxlen = maxlen) # egitim verisini max uxunluga gore ayarla
x_test = pad_sequences(x_test, maxlen=maxlen)

word_index = imdb.get_word_index() # imdb'deki kelimelerin indexlerini al

# kelime dizinini  geir dondurmek icin ters cevir
reverse_word_index = {index + 3: word for word, index in word_index.items()}
reverse_word_index[0] = "<PAD>" # PAD ile eslestir
reverse_word_index[1] = "<START>"
reverse_word_index[2] = "<UNK>"
reverse_word_index[3] = "<UNUSED>"

# ornek metinleri yazdirmma
def decode_review(encoded_review):
    return " ".join([reverse_word_index.get(i, "?") for i in encoded_review]) # her sayiyi kelimeye cevir

#rastgele 3 ornek yazdiralim
random_indices = np.random.choice(len(x_train), size=3, replace=False)
for i in random_indices:
    print(f"Yorum: {decode_review(x_train[i])}")
    print(f"Etiket: {y_train[i]}")
    print()
# %% Transformer layer
class TransformerBlock(layers.Layer):
    def __init__(self, embed_size, heads, dropout_rate = 0.3):
        
        super(TransformerBlock,self).__init__()
        
        self.attention = layers.MultiHeadAttention(num_heads = heads, key_dim = embed_size)
        self.norm1 = layers.LayerNormalization(epsilon = 1e-6) # 1.normalizasyon katmani
        self.norm2 = layers.LayerNormalization(epsilon = 1e-6) # 2.normalizasyon katmani
        self.feed_forward = models.Sequential([
            layers.Dense(embed_size*4, activation = "relu"),
            layers.Dense(embed_size)]) 
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
        
    def call(self, x, training):
        # dikkat (attention) mekanizmasi
        attention = self.attention(x,x)
        x = self.norm1(x + self.dropout1(attention, training = training))
        feed_forward = self.feed_forward(x)
        return self.norm2(x+ self.dropout2(feed_forward, training = training))
    


# %% Transformer Modeli
class TransformerModel(models.Model):
    
    def __init__(self, num_layers, embed_size, heads, input_dim, output_dim, dropout_rate=0.1):
        super(TransformerModel,self).__init__()
        
        self.embedding = layers.Embedding(input_dim=input_dim, output_dim=output_dim)
        self.transformer_blocks = [TransformerBlock(embed_size, heads, dropout_rate) for _ in range(num_layers)]
        self.global_avg_pooling = layers.GlobalAveragePooling1D()
        self.dropout = layers.Dropout(dropout_rate)
        self.fc = layers.Dense(output_dim, activation ="sigmoid")
        
    def call(self, x, training):
        x = self.embedding(x)
        for transformer in self.transformer_blocks:
            x= transformer(x, training)
        x = self.global_avg_pooling(x)
        x = self.dropout(x,training = training)
        
        return self.fc(x)

# %% Model Training

#hyperparameter tanimla
num_layers = 4 # kullanilacak transformer katmanlari
embed_size = 64
num_heads = 4 # multi-head sayisi
input_dim = max_features
output_dim = 1 # ikili siniflandirma 1-0
dropout_rate = 0.1

# modeli olusturma
model= TransformerModel(num_layers, embed_size, num_heads, input_dim, output_dim, dropout_rate)

# modeli bir giris verisi ile cagirarak insa etme
model.build(input_shape=(None, maxlen))

# compile
model.compile(optimizer="adam",loss="binary_crossentropy", metrics=["accuracy"])

model.summary()

# model egitimi
history = model.fit(x_train, y_train,epochs =5,batch_size=256, validation_data=(x_test,y_test))
# %% Model Testing

plt.figure()
plt.subplot(1,2,1)
plt.plot(history.history["loss"], label = "Training Loss")
plt.plot(history.history["val_loss"], label = "Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history["accuracy"], label = "Training Accuracy")
plt.plot(history.history["val_accuracy"], label = "Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.grid(True) 
plt.legend()

plt.tight_layout()
plt.show()

# kullanici metin girdisi al
def predict_sentiment(model, text, word_index, maxlen):
    
    # metni imdb formatinda sayisala cevir
    encoded_text = [word_index.get(word,0) for word in text.lower().split()] # kelimleri sayiya cevir
    padded_text = pad_sequences([encoded_text], maxlen=maxlen) 
    prediction= model.predict(padded_text) # model ile tahmin yap
    return prediction[0][0]

# imdb veri setindeki kelime dizini
word_index = imdb.get_word_index()

# kullanicidan metin al
user_input = input("Bir film yorumu girin: ")
sentiment_score = predict_sentiment(model, user_input, word_index, maxlen)

if sentiment_score > 0.5:
    print(f"Tahmin sonucu olumlu -> skor: {sentiment_score}")
else:
    print(f"Rahmin sonucu olumsuz -> skor: {sentiment_score}")









