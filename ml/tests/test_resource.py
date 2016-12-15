import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
import unittest
import shutil
import time
from ml.resource import Resource
from ml.model import NumberRecognizeNN


class TestResource(unittest.TestCase):
    TEST_DIR = ""

    @classmethod
    def setUpClass(cls):
        path = os.path.join(os.path.dirname(__file__), "./test_resource")
        if not os.path.isdir(path):
            os.mkdir(path)
        cls.TEST_DIR = path

    @classmethod
    def tearDownClass(cls):
        if os.path.isdir(cls.TEST_DIR):
            shutil.rmtree(cls.TEST_DIR)
    
    def test_normalization_parameter(self):
        means = (0.0, 1.0, 0.2)
        stds = (0.5, 0.2, 3.0)
        r = Resource(self.TEST_DIR)
        r.save_normalization_params(means, stds)
        self.assertTrue(os.path.isfile(r.param_file))
        loaded_means, loaded_stds = r.load_normalization_params()
        for i in range(len(means)):
            self.assertTrue(means[i] - loaded_means[i] < 1e-10)
            self.assertTrue(stds[i] - loaded_stds[i] < 1e-10)
    
    def test_save_data(self):
        r = Resource(self.TEST_DIR)
        data_file = self.TEST_DIR + "/data_file.txt"
        data1 = ["0"] + ["0"] * 6400  # label + feature
        data2 = ["9"] + ["1"] * 6400  # label + feature
        r.save_data(data_file, data1)
        r.save_data(data_file, data2)

        x, y = r.load_data(data_file)
        self.assertEqual(2, len(x))
        self.assertEqual(2, len(y))
        self.assertEqual(0, y[0])
        self.assertEqual(9, y[1])
        self.assertEqual(0, x[0][0])
        self.assertEqual(1, x[1][0])

    def test_model(self):
        model = NumberRecognizeNN(10, 10)
        r = Resource(self.TEST_DIR)
        r.save_model(model)
        time.sleep(1)
        r.save_model(model)
        r.load_model(model)


if __name__ == "__main__":
    unittest.main()
