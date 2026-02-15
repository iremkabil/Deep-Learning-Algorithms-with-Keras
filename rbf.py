# %% load_dataset
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer # etiketleri one_hot encoding formatina donusturme  
from sklearn.preprocessing import StandardScaler # veriyi standardize eder, 0 ortalamali 1 std dagilim
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Layer, Dense, Flatten
from tensorflow.keras import backend as K # keras backend API (tensor islemlerini gerceklestirir)

import warnings
warnings.filterwarnings("ignore")

iris = load_iris()

X = iris.data
y= iris.target

# one-hot encoding
label_binarizer = LabelBinarizer() # one-hot encoding formatina donusturen format
y_encoded = label_binarizer.fit_transform(y)

# normalization
scaler = StandardScaler() 
X_scaled = scaler.fit_transform(X)

# train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded,test_size =0.2, random_state=42) 


# %% radyal temelli fonksiyon (RNF) katmani tanimlama
class RBFLayer(Layer): # RBFLayer, Keras'in layer sinifindan miras (inheritance) alir.
    def __init__(self, units, gamma, **kwargs):
        """
            constructor,
            katmanin genel ozelliklerini baslatmak icin gereklidir
        """
        super(RBFLayer, self).__init__(**kwargs) # layer sinifinin init metodunu cagirir, katmanin genel ozelliklerini baslatmak icin gerekli
        self.units = units # rbf katmaninda gizli noron sayisi
        self.gamma = K.cast_to_floatx(gamma) # rbf fonk.yayilim parametresi, rnf duyarliliği diyebiliriz.
    
    def build(self, input_shape):
        """
            build metodu katmanin agirliklarini tanimlar
            bu metot, keras tarafindan katman ilk defa bir input aldiginda otomatik olarak cagirilir
        """
        # add_weight = kerasta egitilebilecek agirliklari tanimlamak icin kullanilir.
        self.mu = self.add_weight(name = "mu",
                                  shape = (int(input_shape[1]), self.units), # shape = agirliklarin boyutunu tanimlar, input_shape[1] = giris verisinin boyutu, self_units=merkezlerin sayisi
                                  initializer = "uniform", # agirliklarin baslangic degeri belirlenir.
                                  trainable = True # agirliklar egitilebilir
                                  )
        super(RBFLayer,self).build(input_shape) # layer sinifinin build metodu cagirilarak katmanin insasi tamamlanir
        
    
    def call(self,inputs):
        """
            katman cagirildiginda (yani forward propagation sirasinda) calisir,
            bu fonk. girdiyi alir, ciktiyi hesaplar.
        """
        diff = K.expand_dims(inputs) - self.mu # K.expand_dims(inputs) girdiye bir boyut ekler
        l2 = K.sum(K.pow(diff, 2),axis = 1) # K.pow(diff, 2)
        res = K.exp(-1 * self.gamma*l2) # K.exp: rbf = exp(-gamma*l2)
        return res
    
    def compute_output_shape(self, input_shape):
        """
            bu metot katmanin ciktisinin sekli hakkinda bilgi verir.
            keras yardimci fonk.larindan birisidir.

        """
        return (input_shape[0],self.units) #ciktinin sekli (num_samples, num_units). input_shape[0] = sapmple sayisi


# %% build, compile and train model 

def build_model():
    model=Sequential()
    model.add(Flatten(input_shape=(4,))) # giris verisini duzlestir,
    model.add(RBFLayer(10, 0.5)) # rbf katmani ekle, 10 noron, gamma = 0.5
    model.add(Dense(3, activation = "softmax")) # output katmani 3 (sinif sayisi > 2) sinif oldugu icin softmax
    
    # compile
    model.compile(optimizer = "adam",
                  loss ="categorical_crossentropy",
                  metrics = ["accuracy"]
                  )
    return model

# model olustur
model = build_model()

history = model.fit(X_train, y_train,
                    epochs = 250,
                    batch_size =4,
                    validation_split = 0.3,
                    verbose = 1)



# %% model evaluation

# test veri seti ile
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test loss: {loss:.4f}, Test Accuracy:{accuracy:.4f}")

# history: loss ve accuracy visualization
plt.figure()
plt.subplot(1,2,1)
plt.plot(history.history["loss"], label = "Training Loss")
plt.plot(history.history["val_loss"], label = "validation loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history["accuracy"], label = "Training Accuracy")
plt.plot(history.history["val_loss"], label = "validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()











