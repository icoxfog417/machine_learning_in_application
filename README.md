# Machine Learning In Application

Practical implemantation of Machine Learning in the Application.  

## Architecture

![architecture.PNG](./docs/architecture.PNG)

* Model: Machine Learning Model
* Trainer: Training the model. So training process (loss, optimizer) is separated from Model.
* Model API: Interface between the Model and Application.
* DataProcessor: Load the data and preprocess it. It is used in Trainer and ModelAPI.
* Resource: It manages the parameters for Trainer, Model and DataProcessor.

## Demo Application

![top.PNG](./docs/top.PNG)

handwritten number recognizer by Chainer.

[![Deploy](https://www.herokucdn.com/deploy/button.png)](https://heroku.com/deploy)

If you want to create the application from command line, please use the below command.

```
heroku create --buildpack https://github.com/kennethreitz/conda-buildpack.git
```
