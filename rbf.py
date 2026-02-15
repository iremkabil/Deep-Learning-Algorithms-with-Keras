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
y_encoding = label_binarizer.fit_transform(y)

# normalization
scaler = StandardScaler() 
X_scaled = scaler.fit_transform(X)

# train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size =0.2, random_state=42) 


# %% radyal temelli fonksiyon (RNF) katmani tanimlama
class RBFLayer(Layer): # RBDLayer, Keras'in layer sinifindan miras (inheritance) alir.
    def __init__(self, units, gamma, **kwargs):
        """
            constructor,
            katmanin genel ozelliklerini baslatmak icin gereklidir
        """
        super(RBFLayer, self).__init__(**kwargs) # layer sinifinin init metodunu cagirir, katmanin genel ozelliklerini baslatmak icin gerekli
        self.units = units # rbf katmaninda gizli noron sayisi
        self.gamma = gamma # rbf fonk.yayilim parametresi, rnf duyarliliği diyebiliriz.
        
        pass
    
    def build(self, input_shape):
        """
            build metodu katmanin agirliklarini tanimlar
            bu metot, keras tarafindan katman ilk defa bir input aldiginda otomatik olarak cagirilir
        """
        pass
    
    def call(self,inputs):
        """
            katman cagirildiginda (yani forward propagation sirasinda) calisir,
            bu fonk. girdiyi alir, ciktiyi hesaplar.
        """
        def compute_output_shape(self, input_shape):
            """
                bu metot katmanin ciktisinin sekli hakkinda bilgi verir.
                keras yardimci fonk.larindan birisidir.

            """





# %% build, compile and train model 



# %% model evaluation

# test veri seti ile








# history: loss ve accuracy visualization


