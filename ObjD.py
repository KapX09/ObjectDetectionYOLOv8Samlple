# !pip install ultralytics opencv-python matplotlib
# To install Libraries

# Upload photos
from google.colab import files
uploaded = files.upload()

# Libraries
from ultralytics import YOLO
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# Load model (YOLOv8 pretrained)
model = YOLO('yolov8n.pt')  # You can use yolov8s.pt or yolov8m.pt for better accuracy
# Set image path
image_path = next(iter(uploaded))  # Get uploaded file name
results = model(image_path)

# Get image shape
img = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w, _ = img.shape
# Show predictions
for result in results:
    print(f"image 1/1 {image_path}: {h}x{w} {len(result.boxes)} objects, {result.speed['inference']:.1f}ms")
    for box in result.boxes:
        cls = int(box.cls[0])
        prob = float(box.conf[0])
        label = model.names[cls]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        print(f"Object type: {label}")
        print(f"Coordinates: [{x1}, {y1}, {x2}, {y2}]")
        print(f"Probability: {round(prob, 2)}")
        print('---')

# Show image with detections
annotated_img = results[0].plot()
plt.imshow(annotated_img)
plt.axis('off')
plt.title('YOLOv8 Detection')
plt.show()

# # Optional: Save
# # Save the image with detections
# output_path = 'detected_output.jpg'
# cv2.imwrite(output_path, cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))
# print(f"Image saved as: {output_path}")

# # To Download the save image
# from google.colab import files
# files.download(output_path)
