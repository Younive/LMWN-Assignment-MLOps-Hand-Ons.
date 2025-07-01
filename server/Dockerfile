# Use an official Python runtime as a parent image.
FROM python:3.10-slim

# Set the working directory in the container to /app
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# CORRECTED PATH: Copy the requirements file from the server directory
COPY server/requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# CORRECTED PATH: Copy the 'app' directory from inside the 'server' directory
COPY server/app ./app

# Copy the Gunicorn configuration file into the working directory
COPY server/gunicorn_conf.py .

# CORRECTED PATH: Now this works because 'model' is inside the build context
COPY model/model.pkl ./model/model.pkl

# Expose the port the app runs on.
EXPOSE 8000

# Define the command to run your app using uvicorn
CMD ["gunicorn", \
     "-c", "gunicorn_conf.py", \
     "--workers", "8", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--timeout", "90", \
     "--bind", "0.0.0.0:8000", \
     "app.main:app"]