# Grace Hopper + Sagemaker Example

This repo contains two seperate examples

1. A simple example of using SageMaker to train a model using Sagemaker training jobs
2. A python based example to run training on a grace hopper instnace


## SageMaker Example
This first thing you want to do is build a base container for machine learning. Something that can be updated ex weekly. 

```Dockerfile
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install dependencies
RUN apt-get update && \
    apt-get install -y ffmpeg libsm6 libxext6 && \
    apt-get clean
```

Push the base container to ECR for x86. You can also build for arm64 and push to ECR if you are using for example a Graviton instance. 
```bash 
docker build . -t 12345678910.dkr.ecr.us-east-1.amazonaws.com/base/machine-learning:latest --platform linux/amd64
```

Build the training container using the base container as the base for the new image

```Dockerfile
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
```

Build and push the container to ECR using platform command for non arm systems container
```bash 
docker build . -t 12345678910.dkr.ecr.us-east-1.amazonaws.com/yolov11:latest   --platform linux/amd64
```

Run the training job using the SageMaker SDK
1. Modify the .env.train file with the correct values for your environment
```python
# Replace with your bucket and dataset paths
import os
import sagemaker
from dotenv import load_dotenv


load_dotenv(dotenv_path=".env.train")

bucket = os.getenv("S3_BUCKET", "haston-home-machine-learning")
dataset_path = os.getenv("DATASET_PATH", "uno-for-twitch")
output_path = f"s3://{bucket}/output/{dataset_path}"
# Retrieve YOLOv10 image URI
image_uri = os.getenv("IMAGE_URI")
# Define SageMaker role
role = os.getenv("SAGEMAKER_ROLE") 
instance_type = os.getenv("INSTANCE_TYPE", "ml.p3.2xlarge")
spot_instance = os.getenv("SPOT_INSTANCE", True)
if isinstance(spot_instance, str):
    spot_instance = bool(spot_instance)
# Create SageMaker Estimator
estimator = sagemaker.estimator.Estimator(
    image_uri=image_uri,
    role=role,
    instance_count=1,
    instance_type=instance_type,
    output_path=output_path,
    use_spot_instances=spot_instance,
    **({"max_wait": 14400, "max_run": 14400} if spot_instance else {}),
    environment={
        "EPOCS": "100",
        "PATIENCE": "50",
    }

)

# Define input channels
inputs = {
    "all": f"s3://{bucket}/{dataset_path}/",
  
}

# Launch the training job
estimator.fit(inputs)
```
Run the training job
```bash
python training-job.py
```

## Grace Hopper Example
    
The second example is a simple python based example to run training on a grace hopper instance.

1. Stand up a gracehopper instance. Lambdalabs has one for $1.25 and hour
2. Run rsync change the IP address to the IP address of your instance
```bash
rsync -av --delete . ubuntu@192.222.1.1:/home/ubuntu/fs-lambda/
```

3. SSH into the instance and run the training script
```bash
ssh ubuntu@192.222.1.1
cd /home/ubuntu/fs-lambda
```
Download dataset from https://public.roboflow.com/object-detection/uno-cards into /home/ubuntu/fs-lambda/uno-experiment/data
```bash
sudo sh setup_gh.sh
python train_gh.py
```

