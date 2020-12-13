FROM python:3
RUN apt−get −y update
RUN apt−get install −y pip3 build−essential
COPY . .
RUN pip3 install −r requirements.txt
ENTRYPOINT ["python3"]
CMD ["app.py"]