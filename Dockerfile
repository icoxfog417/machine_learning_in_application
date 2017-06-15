FROM continuumio/miniconda3

# For Slackbot Dockerfile

ENTRYPOINT []
CMD [ "/bin/bash" ]

# Remove (large file sizes) MKL optimizations.
RUN conda install -y nomkl
RUN conda install -y numpy scipy scikit-learn cython

ADD ./requirements.txt /tmp/requirements.txt
RUN pip install -qr /tmp/requirements.txt

ADD . /opt/ml_in_app
WORKDIR /opt/ml_in_app

CMD python run_application.py
