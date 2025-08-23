FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "i2text.py", "--server.port=8501", "--server.address=0.0.0.0"]
