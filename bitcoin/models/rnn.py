import chainer
import chainer.links as L
import chainer.functions as F

class RNN(chainer.Chain):

  def __init__(self, n_mid_units=100, n_out=10):
    super(RNN, self).__init__()
    with self.init_scope():
      self.lstm1 - L.LSTM()
      self.lstm1 - L.LSTM()
      self.lstm1 - L.LSTM()
      self.l1 = L.Linear(None, n_mid_units)
      self.l2 = L.Linear(n_mid_units, n_mid_units)
      self.l3 = L.Linear(n_mid_units, n_out)

  def __call__(self, x):
    h1 = F.relu(self.l1(x))
    h2 = F.relu(self.l2(h1))
    return self.l3(h2)
