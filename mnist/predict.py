from pathlib import Path

import matplotlib.pyplot as plt
from chainer import serializers
from chainer.datasets import mnist
from models.mlp import MLP

def main():
  model_path = Path('./mnist.model')

  if not model_path.is_file():
    print('Model not found error.')
    exit(-1)

  model = MLP()
  serializers.load_npz(str(model_path), model)

  _, test = mnist.get_mnist(withlabel=True, ndim=1)
  x, _ = test[0]

  plt.imshow(x.reshape(28, 28), cmap='gray')
  plt.show()

  print(f'元の形: {x.shape}', end=' -> ')

  x = x[None, ...]

  print(f'ミニバッチの形にしたあと: {x.shape}')

  y = model(x).array

  print(f'予測結果: {y.argmax(axis=1)[0]}')

if __name__ == '__main__':
  main()