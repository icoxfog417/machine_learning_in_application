import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
import unittest
import numpy as np
from ml.model import NumberRecognizeNN



class TestModel(unittest.TestCase):

    def test_forward(self):
        input_size = 100
        output_size = 10
        data_length = 50
        test_data = self.create_test_data(input_size, data_length)

        model = NumberRecognizeNN(input_size, output_size)
        output = model(test_data)
        self.assertEqual((data_length, output_size), output.data.shape)

    def create_test_data(self, input_size, length):
        input = np.random.rand(length, input_size).astype(np.float32)
        return input


if __name__ == "__main__":
    unittest.main()
