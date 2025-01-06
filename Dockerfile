# Use the base image with ECR for the machine learning image
ARG ACCOUNT_ID
FROM ${ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/base/machine-learning:latest

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt . 
RUN pip install --no-cache-dir -r requirements.txt

COPY train.py /opt/ml/code/train.py
COPY yolo11m.pt /opt/ml/code/yolo11m.pt

# Set the entrypoint
ENTRYPOINT ["python", "/opt/ml/code/train.py"]
