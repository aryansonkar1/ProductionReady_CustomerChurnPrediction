FROM python:3.12-slim

WORKDIR /app

# Install dependencies (pinned to match trained model versions exactly)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only what is needed — no data/ folder (gitignored)
COPY src/ /app/src/
COPY configs/ /app/configs/
COPY models/ /app/models/
COPY streamlit_app.py /app/streamlit_app.py
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Render maps $PORT externally to 8501
EXPOSE 8501

CMD ["/app/entrypoint.sh"]
