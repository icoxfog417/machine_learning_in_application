import chainer
import chainer.functions as F
import chainer.links as L


class NumberRecognizeNN(chainer.Chain):

    def __init__(self, input_size, output_size, hidden_size=200, layer_size=3):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.layer_size = layer_size
        super(NumberRecognizeNN, self).__init__(
            l1=L.Linear(self.input_size, hidden_size),
            l2=L.Linear(hidden_size, hidden_size),
            l3=L.Linear(hidden_size, self.output_size),
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        o = F.sigmoid(self.l3(h2))
        return o
