import numpy as np
from sklearn.model_selection import train_test_split
import chainer
from chainer.functions.loss import softmax_cross_entropy
from chainer.functions.evaluation import accuracy
from ml.data_processor import DataProcessor


class Trainer():

    def __init__(self, model, resource):
        self.model = model
        self.resource = resource
    
    def train(self, data, target, batch_size=100, epoch=5, test_size=0.3, report_interval_epoch=1):
        dp = DataProcessor()
        dp.set_normalization_params(data)
        self.resource.save_normalization_params(dp.means, dp.stds)
        _data = dp.format_x(data)
        _target = dp.format_y(target)
        train_x, test_x, train_y, test_y = train_test_split(_data, _target, test_size=test_size)

        optimizer = chainer.optimizers.Adam()
        optimizer.use_cleargrads()
        optimizer.setup(self.model)
        loss = lambda pred, teacher: softmax_cross_entropy.softmax_cross_entropy(pred, teacher)
        for x_batch, y_batch, epoch_end in dp.batch_iter(train_x, train_y, batch_size, epoch):
            predicted = self.model(x_batch)
            optimizer.update(loss, predicted, y_batch)
            if epoch_end:
                train_acc = accuracy.accuracy(predicted, y_batch)
                predicted_to_test = self.model(test_x)
                test_acc = accuracy.accuracy(predicted_to_test, test_y)
                print("train accuracy={}, test accuracy={}".format(train_acc.data, test_acc.data))
                self.resource.save_model(self.model)
