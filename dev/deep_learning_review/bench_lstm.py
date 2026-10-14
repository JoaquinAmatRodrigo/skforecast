import os
import time
os.environ["KERAS_BACKEND"]="torch"
import numpy as np
import torch
import keras
torch.set_num_threads(4)
n, L, H = 11000, 48, 24
X = np.random.rand(n, L, 1).astype("float32"); Y = np.random.rand(n, H).astype("float32")
# keras
m = keras.Sequential([keras.Input((L,1)), keras.layers.LSTM(100), keras.layers.Dense(64, activation="relu"), keras.layers.Dense(H)])
m.compile("adam","mse")
m.fit(X[:512],Y[:512],epochs=1,batch_size=256,verbose=0)
t=time.time(); m.fit(X,Y,epochs=1,batch_size=256,verbose=0); print(f"keras(torch) LSTM epoch: {time.time()-t:.1f}s")
# native torch
class Net(torch.nn.Module):
    def __init__(s):
        super().__init__(); s.l=torch.nn.LSTM(1,100,batch_first=True); s.d=torch.nn.Sequential(torch.nn.Linear(100,64),torch.nn.ReLU(),torch.nn.Linear(64,H))
    def forward(s,x): return s.d(s.l(x)[0][:,-1])
net=Net(); opt=torch.optim.Adam(net.parameters()); Xt=torch.tensor(X); Yt=torch.tensor(Y)
t=time.time()
for i in range(0,n,256):
    opt.zero_grad(); loss=((net(Xt[i:i+256])-Yt[i:i+256])**2).mean(); loss.backward(); opt.step()
print(f"native torch LSTM epoch: {time.time()-t:.1f}s")
