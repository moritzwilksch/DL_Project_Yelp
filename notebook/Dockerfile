FROM jupyter/datascience-notebook
WORKDIR /app
COPY . /app

RUN apt -y install python3.7
RUN apt -y install python3-pip
RUN python3.7 -m pip install --upgrade pip

RUN pip install -r requirements.txt