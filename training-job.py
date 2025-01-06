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
