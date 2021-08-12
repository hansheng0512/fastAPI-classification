FROM python:3.8.5-slim

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY ./server .

CMD ["python", "main.py"]