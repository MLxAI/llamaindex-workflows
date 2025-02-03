FROM python:3.11
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . /app
ENV OPENAI_API_KEY=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
ENV QDRANT_HOST=10.0.78.175