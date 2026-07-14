FROM python:3.11.9-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Download tokenizer during build
RUN python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('bert-base-uncased')"

COPY . .

CMD ["sh", "-c", "uvicorn backend.app:app --host 0.0.0.0 --port ${PORT}"]