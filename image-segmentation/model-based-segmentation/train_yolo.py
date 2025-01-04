from ultralytics import YOLO

model = YOLO('yolov9s.pt')

# Train the model
model.train(data='./dataset.yaml', epochs=5, imgsz=480)
