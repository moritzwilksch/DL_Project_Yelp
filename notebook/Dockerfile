FROM python:3.7
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
CMD ["jupyter", "notebook",  "--ip=0.0.0.0", "--port=8080"]