import numpy as np
from ml.model import NumberRecognizeNN
from ml.data_processor import DataProcessor


class ModelAPI():

    def __init__(self, resource):
        self.resource = resource
        self.model = NumberRecognizeNN(resource.INPUT_SIZE, resource.OUTPUT_SIZE)
        resource.load_model(self.model)

        means, stds = resource.load_normalization_params()
        self.dp = DataProcessor(means, stds)

    def predict(self, data):
        _data = data
        if isinstance(data, (tuple, list)):
            _data = np.array([data], dtype=np.float32)

        f_data = self.dp.format_x(_data, size=self.resource.INPUT_SIZE)
        predicted = self.model(f_data)
        number = np.argmax(predicted.data, axis=1)
        return number
