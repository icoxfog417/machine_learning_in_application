import os
import json
from datetime import datetime
import numpy as np
from chainer import serializers
from ml.data_processor import DataProcessor


class Resource():
    INPUT_SIZE = 64  # 8 x 8 image size
    OUTPUT_SIZE = 10  # 10 classification

    def __init__(self, root=""):
        self.root = root if root else os.path.join(os.path.dirname(__file__), "./store")
        self.model_path = os.path.join(self.root, "./model")
        self.param_file = os.path.join(self.root, "./params.json")
    
    def save_normalization_params(self, means, stds):
        to_list = lambda ls: ls if isinstance(ls, (tuple, list)) else ls.tolist()
        params = {
            "means": to_list(means),
            "stds": to_list(stds)
        }
        serialized = json.dumps(params)
        with open(self.param_file, "wb") as f:
            f.write(serialized.encode("utf-8"))

    def load_normalization_params(self):
        loaded = {}
        if not os.path.isfile(self.param_file):
            raise Exception("Normalization parameter file does not exist.")

        with open(self.param_file, "rb") as f:
            loaded = json.loads(f.read().decode("utf-8"))
        
        to_array = lambda x: np.array([float(_x) for _x in x], dtype=np.float32)

        return to_array(loaded["means"]), to_array(loaded["stds"])
    
    def load_training_data(self):
        from sklearn.datasets import load_digits
        # predifine set is from scikit-learn dataset
        # http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html

        digits = load_digits()
        x = digits.data
        y = digits.target

        return x, y

    def save_data(self, path, data):
        with open(path, "ab") as f:
            label = int(data[0])
            features = [float(d) for d in data[1:]]
            if len(features) > self.INPUT_SIZE:
                dp = DataProcessor()
                features = dp.adjust(np.array([features]), self.INPUT_SIZE).tolist()[0]
            elif len(features) < self.INPUT_SIZE:
                raise Exception("Size mismatch when saving the data.")
            line = "\t".join([str(e) for e in [label] + features]) + "\n"
            f.write(line.encode("utf-8"))

    def load_data(self, path):
        x = []
        y = []
        with open(path, mode="r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                label, features = self.read_data(line)
                x.append(features)
                y.append(label)
        x = np.array(x, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        return x, y

    def read_data(self, line):
        elements = line.split("\t")
        label = int(elements[0])
        features = [float(e) for e in elements[1:]]
        return label, features

    def save_model(self, model):
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        timestamp = datetime.strftime(datetime.now(), "%Y%m%d%H%M%S")
        model_file = os.path.join(self.model_path, "./" + model.__class__.__name__.lower() + "_" + timestamp + ".model")
        serializers.save_npz(model_file, model)
    
    def load_model(self, model):
        if not os.path.exists(self.model_path):
            raise Exception("model file directory does not exist.")

        suffix = ".model"
        keyword = model.__class__.__name__.lower()
        candidates = []
        for f in os.listdir(self.model_path):
            if keyword in f and f.endswith(suffix):
                candidates.append(f)
        candidates.sort()
        latest = candidates[-1]
        #print("targets {}, pick up {}.".format(candidates, latest))
        model_file = os.path.join(self.model_path, latest)
        serializers.load_npz(model_file, model)
