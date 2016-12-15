import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
import math
import unittest
import numpy as np
from ml.resource import Resource
from ml.data_processor import DataProcessor


class TestDataProcessor(unittest.TestCase):
    
    def test_format_x(self):
        means = np.array([0, 0.1, 0.2])
        stds = np.array([1, 1.5, 0.5])
        dp = DataProcessor(means=means, stds=stds)
        data = np.array([[1, 2, 3], [4, 5, 6]])
        x = dp.format_x(data)
        _x = (data - means) / stds
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                self.assertEqual(x[i][j], _x[i][j])

    def test_format_x_resize(self):
        dp = DataProcessor()
        data = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]])
        x = dp.format_x(data, size=4)
        v = x[0].tolist()
        self.assertEqual(v[0], 6)
        self.assertEqual(v[1], 8)
        self.assertEqual(v[2], 14)
        self.assertEqual(v[3], 16)

    def test_batch_iter(self):
        batch_size = 10
        dp = DataProcessor()
        r = Resource()
        train_x, train_y = r.load_training_data()
        batch_count = math.ceil(len(train_y) / batch_size)

        i = 0
        for x_batch, y_batch, epoch_end in dp.batch_iter(train_x, train_y, batch_size):
            self.assertEqual(batch_size, len(x_batch))
            self.assertEqual(batch_size, len(y_batch))
            if i < batch_count - 1:
                self.assertFalse(epoch_end)
            else:
                self.assertTrue(epoch_end)
            i += 1
        self.assertEqual(i, batch_count)


if __name__ == "__main__":
    unittest.main()

