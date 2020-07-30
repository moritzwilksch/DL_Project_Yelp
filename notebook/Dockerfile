FROM python:3.7
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]