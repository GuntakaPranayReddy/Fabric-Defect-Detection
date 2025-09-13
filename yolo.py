from ultralytics import YOLO
model=YOLO('best.pt')
metrics=model.predict('images',save=True)