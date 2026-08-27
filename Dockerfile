FROM python:3.10-slim

WORKDIR /app

# Install dependencies (pinned versions match the trained model)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install streamlit==1.31.0

# Copy everything needed by Streamlit + the model pipeline
COPY src/ /app/src/
COPY configs/ /app/configs/
COPY models/ /app/models/
COPY data/ /app/data/
COPY streamlit_app.py /app/streamlit_app.py

# Copy entrypoint
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Render will inject $PORT and map it externally
EXPOSE 8501

CMD ["/app/entrypoint.sh"]
