import os
from ultralytics import YOLO
import cv2
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logger
logger = logging.getLogger(__name__)
log_level = os.getenv("LOG_LEVEL", "ERROR").upper()
logging.basicConfig(level=getattr(logging, log_level, logging.INFO))

# Environment Variables
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.5))
INFERENCE_DEVICE = os.getenv("INFERENCE_DEVICE", "cpu")  # Default to CPU
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output_frames")
INPUT_DIR = os.getenv("INPUT_DIR", "process")

# Load YOLO model
try:
    model = YOLO("last_llabs.pt")  # Replace with your model file
    logger.info("YOLO model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading YOLO model: {e}")
    exit()

# Parse target classes from environment variable
notify_if = os.getenv("NOTIFY_IF", "0,1,10,11,12,13,14,2,3,4,5,6,7,8,9")
target_classes = [cls.strip().lower() for cls in notify_if.split(",")]
logger.info(f"Target classes for detection: {target_classes}")

# Create output directory for processed images
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Function to process a single image
def process_image(image_path):
    try:
        # Read the image
        frame = cv2.imread(image_path)
        if frame is None:
            logger.error(f"Error: Unable to read image {image_path}.")
            return

        # Perform inference
        results = model.predict(frame, device=INFERENCE_DEVICE, conf=CONFIDENCE_THRESHOLD)

        # Check for target class detection and annotate
        target_detected = False
        for box in results[0].boxes:
            class_id = int(box.cls)
            class_name = results[0].names[class_id].lower()
            probability = box.conf.item()
            logger.info(f"Detected: {class_name} with probability: {probability:.2f}")
            if class_name in target_classes:
                target_detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36, 255, 12), 2)

        # Save the annotated image if any target was detected
        if target_detected:
            output_path = os.path.join(OUTPUT_DIR, f"detected_{os.path.basename(image_path)}")
            cv2.imwrite(output_path, frame)
            logger.info(f"Target class detected! Saved annotated image to {output_path}")
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {e}")

# Function to process a folder of images
def process_folder(folder_path):
    if not os.path.exists(folder_path):
        logger.error(f"Error: Folder {folder_path} does not exist.")
        return

    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    logger.info(f"Found {len(image_files)} images in folder {folder_path}.")

    for image_path in image_files:
        logger.info(f"Processing image: {image_path}")
        process_image(image_path)
        # Uncomment the following line to delete processed images
        # os.remove(image_path)

# Process all images in the input folder
process_folder(INPUT_DIR)
