from ultralytics import YOLO
import os 
import torch

# Allow user to overide hyperparameters
EPOCS = int(os.getenv("EPOCS", 300))
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", 640))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", -1))
WORKERS = int(os.getenv("WORKERS", 8))
PATIENCE = int(os.getenv("PATIENCE", 100))
SAGEMAKER_JOB_DIR = os.getenv("SAGEMAKER_JOB_DIR", "/opt/ml/input/data/all")
DATA_YAML_FILE_NAME = os.getenv("DATA_YAML_FILE_NAME", "data.yaml")
LEARNING_RATE_INITIAL = float(os.getenv("LEARNING_RATE_INITIAL", 0.01))
LEARNING_RATE_FINAL = float(os.getenv("LEARNING_RATE_FINAL", 0.01))
CLS = float(os.getenv("CLS", 0.5 ))
BOX = float(os.getenv("BOX", 7.5))
DFL = float(os.getenv("DFL", 1.5))
WEIGHTED_DECAY = float(os.getenv("WEIGHTED_DECAY", 0.0005))
DROPOUT = float(os.getenv("DROPOUT", 0.0))
CLOSE_MOSAIC= int(os.getenv("CLOSE_MOSAIC", 10))
CACHE = bool(os.getenv("CACHE", False))
VALIDATION = bool(os.getenv("VALIDATION", True))


device = "cpu"
if torch.cuda.is_available():
    print("Using GPU")
    device = []
    gpu_num = 0
    gpu_count = torch.cuda.device_count()
    for gpu_num in range(gpu_count-1):
        device.append(gpu_num)
        gpu_num += 1


# Initialize the model 
model=YOLO("/opt/ml/code/yolo11m.pt")

# Train the model
model.train(
    data=f"{SAGEMAKER_JOB_DIR}/{DATA_YAML_FILE_NAME}",  # SageMaker path for input data
    epochs=EPOCS,
    imgsz=IMAGE_SIZE,
    batch=BATCH_SIZE,
    lr0=LEARNING_RATE_INITIAL,  # Default is 0.01. Start with a smaller learning rate to prevent large updates.
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
    project="/opt/ml/model",                      # Root directory for saving results
    device=device  # GPU index
)





