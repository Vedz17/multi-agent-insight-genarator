# 1. Base Image: Hum lightweight Python 3.10 use kar rahe hain
FROM python:3.10-slim

# 2. Container ke andar ek folder bana rahe hain '/app' naam se
WORKDIR /app

# 3. Pehle sirf requirements.txt copy karo (taaki caching fast ho)
COPY requirements.txt .

# 4. Saari libraries install karo (Gunicorn aur Uvicorn bhi add kar rahe hain production ke liye)
RUN pip install --no-cache-dir -r requirements.txt gunicorn uvicorn

# 5. Ab bacha hua saara backend code copy kar lo
COPY . .

# 6. Container ka port 8000 open karo jahan FastAPI chalta hai
EXPOSE 8000

# 7. Production Server (Gunicorn) start karne ka command
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:8000", "--workers", "2"]