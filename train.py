import os
import argparse
from ml.model import NumberRecognizeNN
from ml.data_processor import DataProcessor
from ml.trainer import Trainer
from ml.resource import Resource


def train(data_file, batch_size, epoch, test_size):
    r = Resource()
    dp = DataProcessor()
    model = NumberRecognizeNN(Resource.INPUT_SIZE, Resource.OUTPUT_SIZE)
    try:
        dp.means, dp.stds = r.load_normalization_params()
        r.load_model(model)
        print("load the model")
    except Exception as ex:
        print("trained model does not exist.")

    x = None
    y = None
    if data_file:
        x, y = r.load_data(data_file)
    else:
        x, y = r.load_training_data()
    
    trainer = Trainer(model, r)
    print("begin training")
    trainer.train(x, y, batch_size=batch_size, epoch=epoch, test_size=test_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Model")
    parser.add_argument("--data", help="training file", default="")
    parser.add_argument("--batch_size", help="batch size", default=100, type=int)
    parser.add_argument("--epoch", help="epoch size", default=5, type=int)
    parser.add_argument("--test_size", help="test_size", default=0.3, type=float)
    args = parser.parse_args()

    train(args.data, args.batch_size, args.epoch, args.test_size)
