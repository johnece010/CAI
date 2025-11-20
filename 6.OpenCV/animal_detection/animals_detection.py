from ultralytics import YOLO
import cv2

# Load pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')  # 'n' = nano (fast, lightweight)

# Load image (you can replace with your own)
image_path = "CAI/6.OpenCV/animal_detection/data/cat.jpg"  # Example image file
img = cv2.imread(image_path)

# Run detection
results = model(image_path)

# Display results
for result in results:
    boxes = result.boxes
    for box in boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        conf = box.conf[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        # Draw boxes
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(img, f'{label} {conf:.2f}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

# Show output
cv2.imshow("Animal Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
