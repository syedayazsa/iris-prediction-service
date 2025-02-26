# Use slim python image as base
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy only the necessary files
COPY requirements.txt .
COPY src/ src/
COPY src/models/ src/models/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the app runs on
EXPOSE 8000

# Command to run the API server using gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "src.serve:app"]