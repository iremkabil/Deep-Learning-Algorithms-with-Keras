# %% deep q learning agent class
import gym # reinforcement learning icin env saglar, gelistirme saglar
import numpy as np
from collections import deque # ajanin bellegini tanimlamak icin gerekli deque veri yapisi
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 
from tensorflow.keras.optimizers import Adam

import random

from tqdm import tqdm # ilerlemeyi gorsellestirmek icin

# dql agent class
class DQLAgent:
    
    def __init__(self,env):  # parametreleri ve hiperparametreleri tanimla, env:cartpole
        
        # cevre gozlem alani (state) boyut sayisi
        self.state_size = env.observation_space.shape[0]
        
        # cevrede bulunan eylem sayisi (ajanin secebilecegi eylem sayisi)
        self.action_size = env.action_space.n
        
        # gelecekteki odullerin indirim orani
        self.gamma = 0.95 
        
        # learning rate ajanin ogrenme hizi
        self.learning_rate = 0.001
        
        # kesfetme orani (epsilon) = 1 olsun maximum kesif
        self.epsilon = 1
        
        # epsilonun her iterasyonda azalma orani (epsilon azaldikca daha fazla ogrenme, daha az kesif)
        self.epsilon_decay = 0.995
        
        # minimum kesfetme orani (epsilon 0.01'in altina inemez)
        self.epsilon_min = 0.01

        # ajanin deneyimleri = bellek
        self.memory = deque(maxlen=1000)
        
        # derin ogrenme modelini insaa et
        self.model = self.build_model()
    
    def build_model(self): # deep q learning sinir agi modelini olustur

        model = Sequential() 
        
        # girdi katmani, 48 noron, relu
        model.add(Dense(48, input_dim = self.state_size, activation ="relu"))
        
        # 24 noronlu 2. gizli katman
        model.add(Dense(24,activation = "relu"))
        
        # output layer
        model.add(Dense(self.action_size, activation ="linear"))
        
        # compile model
        model.compile(loss = "mse", optimizer = Adam(lr = self.learning_rate))
        
        return model
    
    def remember(self, state, action, reward, next_state, done): # ajanin deneyimlerini bellek veri yapisina kaydet
    
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state): # ajanimiz eylem secebilecek
    
        # eger rastgele uretilen sayi epsilondan kucukse rastgele eylem secilir (kesif)
        if random.uniform(0.1) <= self.epsilon:
            return env.action_space.sample() # rastgele eylem sec
        # aksi durumda model tarafindan tahmin edilen degerlere gore en iyi eylem secilir
        act_values = self.model.predict(state, verbose = 0)
        
        # en yuksek degere sahip eylemi sec
        return np.argmax(act_values[0])
            
        
    def replay(self, batch_size): # deneyimleri tekrar oynatarak deep q agi egitilir
        
        # bellekte yeterince deneyim yoksa geri oynatma yapilmaz
        if len(self.memory) < batch_size:
            return 
        
        # bellekten rastgele batch size kadar deneyim sec
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            if done: # eger done ise (bitis durum var ise) odulu dogrudan hedef olarak aliriz
                target = reward
            else: 
                target = reward + self.gamma*np.amax(self.model.predict(next_state, verbose=0)[0])
            
            # modelin tahmin ettigi oduller
            train_target = self.model.predict(state,verbose = 0)
            
            # ajanin yaptigi eyleme gore tahmin edilen odulu guncelle
            train_target[0][action] =target
            
            # modeli egit 
            self.model.fit(state,train_target, verbose=0)
            
            
    def adaptiveEGreedy(self): # epsilonun zamanla azalmasi yani kesif ve somuru dengesi
    
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon*self.act()



# %% gym environment'ini kullanarak dql ajanimizi baslatiyoruz(training) 
env = gym.make("CartPole-v1", render_mode="human") # cartpole ortamini baslatiyoruz
agent = DQLAgent(env)

batch_size = 32  # egitim icin minibatch boyutu
episodes = 2 # epoch, simulasyonun oynatilacagi toplam bolum sayisi

for e in tqdm(range(episodes)):
    
    # ortami sifirla ve baslangic durumuna al
    state = env.reset()[0] # ortami sifirlamak
    state = np.reshape(state, [1,4])
    time = 0 # zaman adimini baslat
    
    while true:
        # ajan eylem secer
        
        
        # ajan ortamda bu eylemi uygular ve bu eylem sonucunda next_state, reward,bitis bilgisi(done) alir
        
        
        # yapmis oldugu bu adimi yani eylemi ve bu eylem sonucu env'dan alinan bilgileri kaydeder
        
        # mevcut durumu gunceller
        
        # deneyimlerden yeniden oynatmayi baslatir reply() -> training
        
        # epsilonu set eder
        
        # zaman adimini arttirir
        
        # eger done ise donguden cikar






# %% test





