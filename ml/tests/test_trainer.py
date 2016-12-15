import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
import unittest
import shutil
import numpy as np
from ml.model import NumberRecognizeNN
from ml.data_processor import DataProcessor
from ml.trainer import Trainer
from ml.resource import Resource


class TestTrainer(unittest.TestCase):
    TEST_DIR = ""

    @classmethod
    def setUpClass(cls):
        path = os.path.join(os.path.dirname(__file__), "./test_trainer")
        if not os.path.isdir(path):
            os.mkdir(path)
        cls.TEST_DIR = path

    @classmethod
    def tearDownClass(cls):
        if os.path.isdir(cls.TEST_DIR):
            shutil.rmtree(cls.TEST_DIR)

    def test_train(self):
        model = NumberRecognizeNN(Resource.INPUT_SIZE, Resource.OUTPUT_SIZE)
        r = Resource(self.TEST_DIR)
        trainer = Trainer(model, r)
        dp = DataProcessor()
        data, target = r.load_training_data()
        print("Test Train the model")
        trainer.train(data, target, epoch=5)

    def test_baseline(self):
        from sklearn.svm import SVC
        from sklearn.metrics import accuracy_score
        r = Resource(self.TEST_DIR)
        dp = DataProcessor()
        data, target = r.load_training_data()
        dp.set_normalization_params(data)
        f_data, f_target = dp.format_x(data), dp.format_y(target)

        test_size = 200
        model = SVC()
        model.fit(f_data[:-test_size], f_target[:-test_size])

        predicted = model.predict(f_data[-test_size:])
        teacher = f_target[-test_size:]
        score = accuracy_score(teacher, predicted)
        print("Baseline score is {}".format(score))


if __name__ == "__main__":
    unittest.main()

