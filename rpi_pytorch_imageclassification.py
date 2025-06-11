import time
import torch
import numpy as np
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision import transforms
from picamera2 import Picamera2
import cv2
from PIL import Image

# Load model and weights
weights = MobileNet_V2_Weights.DEFAULT
net = mobilenet_v2(weights=weights)
net.eval()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)

# ImageNet mean and std
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Class labels (ImageNet)
categories = weights.meta["categories"]

# Initialize camera
picam2 = Picamera2()
picam2.preview_configuration.main.size = (224, 224)
picam2.preview_configuration.main.format = "BGR888"
picam2.configure("preview")
picam2.start()
time.sleep(2)

# FPS tracking
frame_count = 0
last_logged = time.time()
fps = 0.0

# Inference + GUI loop
with torch.no_grad():
    while True:
        # Capture image
        image = picam2.capture_array()

        # Copy image for OpenCV display
        display_image = image.copy()

        # Convert to RGB for PIL
        image_rgb = image[:, :, [2, 1, 0]]
        image_pil = Image.fromarray(image_rgb)

        # Preprocess
        input_tensor = preprocess(image_pil)
        input_batch = input_tensor.unsqueeze(0).to(device)

        # Inference
        output = net(input_batch)
        _, predicted = torch.max(output, 1)
        class_idx = predicted.item()
        class_name = categories[class_idx]

        # Calculate FPS
        frame_count += 1
        now = time.time()
        if now - last_logged >= 1.0:
            fps = frame_count / (now - last_logged)
            last_logged = now
            frame_count = 0

        # Overlay prediction and FPS
        cv2.putText(display_image, f"Class: {class_name}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(display_image, f"FPS: {fps:.2f}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Show window
        cv2.imshow("Live Prediction", display_image)

        # Break on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Cleanup
cv2.destroyAllWindows()
picam2.stop()

