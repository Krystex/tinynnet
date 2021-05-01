<p align="center">
  <img src="https://i.imgur.com/xuiyGVu.png" width="400">
</p>


A tiny, proof of concept neural network framework, based on the [Sentdex Tutorials](https://www.youtube.com/watch?v=Wo5dMEP_BbI&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3)

The only dependency is *Numpy*.


```python
from tinynnet import SequentialNet
from tinynnet.layers import Dense
from tinynnet.act import ReLU, Softmax
from tinynnet.tools import to_categorical

net = SequentialNet([
  Dense(2, 3),
  ReLU(),
  Dense(3, 3),
  Softmax()
])
net.forward(data)
```

