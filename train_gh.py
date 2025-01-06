from ultralytics import YOLO
import os
import torch

# Allow user to override hyperparameters
EPOCS = int(os.getenv("EPOCS", 100))
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", 640))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", -1))
WORKERS = int(os.getenv("WORKERS", 32))
PATIENCE = int(os.getenv("PATIENCE", 50))
DATA_DIR = os.getenv("DATA_DIR", "/home/ubuntu/fs-lambda/uno-experiment/data")
DATA_YAML_FILE_NAME = os.getenv("DATA_YAML_FILE_NAME", "data.yaml")
LEARNING_RATE_INITIAL = float(os.getenv("LEARNING_RATE_INITIAL", 0.01))
LEARNING_RATE_FINAL = float(os.getenv("LEARNING_RATE_FINAL", 0.01))
CLS = float(os.getenv("CLS", 0.5))
BOX = float(os.getenv("BOX", 7.5))
DFL = float(os.getenv("DFL", 1.5))
WEIGHTED_DECAY = float(os.getenv("WEIGHTED_DECAY", 0.0005))
DROPOUT = float(os.getenv("DROPOUT", 0.0))
CLOSE_MOSAIC = int(os.getenv("CLOSE_MOSAIC", 10))
CACHE = os.getenv("CACHE", "False").lower() in ["true", "1", "yes"]
VALIDATION = os.getenv("VALIDATION", "True").lower() in ["true", "1", "yes"]

# Determine the device
if torch.cuda.is_available():
    print("Using GPU(s)")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    device = ",".join([str(i) for i in range(torch.cuda.device_count())])  # List all available GPUs
else:
    print("Using CPU")
    device = "cpu"

# Initialize the model
model = YOLO("/home/ubuntu/fs-lambda/yolo11m.pt")

# Train the model
model.train(
    data=f"{DATA_DIR}/{DATA_YAML_FILE_NAME}",  # SageMaker path for input data
    epochs=EPOCS,
    imgsz=IMAGE_SIZE,
    batch=BATCH_SIZE,
    lr0=LEARNING_RATE_INITIAL,  # Default is 0.01
    lrf=LEARNING_RATE_FINAL,
    cls=CLS,
    box=BOX,
    dfl=DFL,
    weight_decay=WEIGHTED_DECAY,
    dropout=DROPOUT,
    close_mosaic=CLOSE_MOSAIC,
    cache=CACHE,
    val=VALIDATION,
    workers=WORKERS,
    patience=PATIENCE,
    project="/home/ubuntu/fs-lambda/data/results",  # Root directory for saving results
    device=device  # Pass GPU(s) or CPU
)




