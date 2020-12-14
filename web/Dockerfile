FROM amd64/python:3

RUN apt-get -y update

RUN apt-get install -y python3-pip build-essential

COPY . .

RUN pip3 install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.12.0-py3-none-any.whl

RUN pip3 install -r requirements.txt

ENTRYPOINT ["python3"]

CMD ["app.py"]