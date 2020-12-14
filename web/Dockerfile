FROM amd64/python:3.8

RUN apt-get -y update

RUN apt-get install -y python3-pip build-essential

COPY . .

RUN pip3 install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow_cpu-2.3.0-cp38-cp38-manylinux2010_x86_64.whl

RUN pip3 install -r requirements.txt

ENTRYPOINT ["python3"]

CMD ["app.py"]