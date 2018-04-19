import random
import numpy as np

import chainer
from chainer.datasets import mnist
from chainer import iterators
import chainer.links as L
import chainer.functions as F
from chainer import optimizers
from chainer.dataset import concat_examples
from chainer import serializers

from models.mlp import MLP

def main():
  random.seed(0)
  np.random.seed(0)

  batch_size = 128
  max_epoch = 10

  train, test = mnist.get_mnist(withlabel=True, ndim=1)

  train_iter = iterators.SerialIterator(train, batch_size)
  test_iter = iterators.SerialIterator(test, batch_size, repeat=False, shuffle=False)

  model = MLP()

  optimizer = optimizers.SGD(lr=0.01)
  optimizer.setup(model)

  while train_iter.epoch < max_epoch:
    train_batch = train_iter.next()
    # ミニバッチ化したtrainデータから入力データと正解ラベルの抽出をする(options: gpu_id=None)
    x, t = concat_examples(train_batch)

    y = model(x)

    loss = F.softmax_cross_entropy(y, t)

    model.cleargrads()
    loss.backward()

    optimizer.update()

    # 1poch辺りの精度検証
    if train_iter.is_new_epoch:
      # 現在のepoch 学習時の誤り率の出力
      print('epoch:{:02d} train_loss:{:.04f} '.format(train_iter.epoch, float(loss.data)), end='')

      test_losses = []
      test_accuracies = []
      while True:
        test_batch = test_iter.next()
        x_test, t_test = concat_examples(test_batch)

        y_test = model(x_test)

        loss_test = F.softmax_cross_entropy(y_test, t_test)
        test_losses.append(loss_test.data)

        accuracy = F.accuracy(y_test, t_test)
        test_accuracies.append(accuracy.data)

        if test_iter.is_new_epoch:
          test_iter.epoch = 0
          test_iter.current_position = 0
          test_iter.is_new_epoch = False
          test_iter._pushed_position = None
          break

      # テストデータにおける誤り率と精度の出力
      print('val_loss:{:.04f} val_accuracy:{:.04f}'.format(np.mean(test_losses), np.mean(test_accuracies)))

  # to_gpuにしている場合はmodel.to_cpu()してcpuの学習モデルにした方が良い
  serializers.save_npz('mnist.model', model)

if __name__ == '__main__':
  main()