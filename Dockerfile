FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install streamlit==1.31.0

# Copy source code
COPY src/ /app/src/
COPY api/ /app/api/
COPY configs/ /app/configs/
COPY models/ /app/models/
COPY streamlit_app.py /app/streamlit_app.py

# Copy entrypoint
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Render exposes $PORT (typically 10000); FastAPI listens there.
# Streamlit runs internally on 8501 (not publicly exposed).
EXPOSE 10000

CMD ["/app/entrypoint.sh"]
