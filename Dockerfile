FROM python:3.13-slim
WORKDIR /app
RUN useradd --create-home app
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
COPY src src
COPY static static
COPY models models
USER app
ENV PYTHONPATH=/app/src OMP_NUM_THREADS=2 PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "exec uvicorn sonarseg.api:app --host 0.0.0.0 --port ${PORT}"]
