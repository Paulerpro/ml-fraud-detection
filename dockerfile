FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential

COPY requirements.txt /app/

RUN python -m pip install --upgrade pip 

RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 8000 8501

CMD ["bash", "-c", "uvicorn api.service:app --host 0.0.0.0 --port 8000 & streamlit run ui/app.py --server.port 8501 --server.address 0.0.0.0"]
