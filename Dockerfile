#FROM ubuntu:23.10
FROM python:3.9.17
COPY ./api/app.py /newdock/
COPY ./requirements.txt /newdock
WORKDIR /newdock

#RUN apt-get update
#RUN apt-get install -y python3 python3-pip

RUN  pip install --upgrade pip
# RUN pip3 install -U scikit-learn --no-cache-dir
# RUN ls
RUN pip3 install -r requirements.txt --no-cache-dir

#ENV FLASK_APP=newdock/hello.py
#CMD ["flask", "run", "--host=0.0.0.0"]
#CMD ["python","-u","digits.py"]
#CMD [ "python3","-u","hello.py", "--host=0.0.0.0" ]
ENV FLASK_APP=api/app
EXPOSE 80
CMD [ "python3","-m","flask", "run", "--host=0.0.0.0", "--port=80"]
