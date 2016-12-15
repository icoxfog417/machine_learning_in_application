import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
import unittest
import shutil
from sklearn.metrics import accuracy_score
from ml.model import NumberRecognizeNN
from ml.model_api import ModelAPI
from ml.trainer import Trainer
from ml.data_processor import DataProcessor
from ml.resource import Resource


class TestModelAPI(unittest.TestCase):
    TEST_DIR = ""

    @classmethod
    def setUpClass(cls):
        path = os.path.join(os.path.dirname(__file__), "./test_model_api")
        if not os.path.isdir(path):
            os.mkdir(path)
        cls.TEST_DIR = path

    @classmethod
    def tearDownClass(cls):
        if os.path.isdir(cls.TEST_DIR):
            shutil.rmtree(cls.TEST_DIR)

    def test_model_api(self):
        model = NumberRecognizeNN(Resource.INPUT_SIZE, Resource.OUTPUT_SIZE)
        r = Resource(self.TEST_DIR)
        trainer = Trainer(model, r)
        dp = DataProcessor()
        data, target = r.load_training_data()
        api_test_size = 200

        print("Train the model for API Test.")
        trainer.train(data[:-api_test_size], target[:-api_test_size], epoch=5)

        model_api = ModelAPI(r)
        predicted = model_api.predict(data[-api_test_size:])
        teacher = target[-api_test_size:]
        score = accuracy_score(teacher, predicted)
        print("Model API score is {}".format(score))


if __name__ == "__main__":
    unittest.main()
