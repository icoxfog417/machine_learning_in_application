import os
import tornado.web
from ml.model_api import ModelAPI
from ml.data_processor import DataProcessor
from ml.resource import Resource


DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/feedbacks.txt")


class IndexHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("index.html", title="title")


class PredictionHandler(tornado.web.RequestHandler):

    def post(self):
        resp = {"result": str(-1)}
        data = self.get_arguments("data[]")

        r = Resource()
        if not os.path.isdir(r.model_path):
            from ml.model import NumberRecognizeNN
            from ml.trainer import Trainer
            model = NumberRecognizeNN(r.INPUT_SIZE, r.OUTPUT_SIZE)
            trainer = Trainer(model, r)
            x, y = r.load_training_data()
            trainer.train(x, y)
        api = ModelAPI(r)

        if len(data) > 0:
            _data = [float(d) for d in data]
            predicted = api.predict(_data)
            resp["result"] = str(predicted[0])

        self.write(resp)


class FeedbackHandler(tornado.web.RequestHandler):

    def post(self):
        data = self.get_arguments("data[]")
        if len(data) > 0:
            r = Resource()
            r.save_data(DATA_PATH, data)
        else:
            result = "feedback format is wrong."

        resp = {"result": ""}
        self.write(resp)


class Application(tornado.web.Application):

    def __init__(self):
        handlers = [
            (r"/", IndexHandler),
            (r"/predict", PredictionHandler),
            (r"/feedback", FeedbackHandler),
        ]

        settings = dict(
            template_path=os.path.join(os.path.dirname(__file__), "templates"),
            static_path=os.path.join(os.path.dirname(__file__), "static"),
            cookie_secret=os.environ.get("SECRET_TOKEN", "__TODO:_GENERATE_YOUR_OWN_RANDOM_VALUE_HERE__"),
            xsrf_cookies=True,
            debug=True,
        )

        super(Application, self).__init__(handlers, **settings)
