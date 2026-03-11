# 1️⃣ Base image
FROM python:3.10-slim

# 2️⃣ Set working directory
WORKDIR /app

# 3️⃣ Copy requirements first (for caching)
COPY requirements.txt .

# 4️⃣ Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5️⃣ Copy entire project
COPY . .

# 6️⃣ Expose port (Render uses 10000)
EXPOSE 10000

# 7️⃣ Run app using gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "app:app"]
