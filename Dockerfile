#FROM ubuntu:23.10
FROM python:3.9.17
COPY ../*.py /newdock/
COPY ../requirements.txt /newdock
WORKDIR /newdock

#RUN apt-get update
#RUN apt-get install -y python3 python3-pip

RUN  pip install --upgrade pip
# RUN pip3 install -U scikit-learn --no-cache-dir
RUN ls
RUN pip3 install -r requirements.txt --no-cache-dir

CMD ["python","-u","digits.py"]