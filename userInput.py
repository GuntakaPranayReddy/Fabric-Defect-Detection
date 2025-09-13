import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import os
import torchvision


# ---------- Load Model ----------
def load_model(model_path, num_classes=2):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

# ---------- Inference on One Image ----------
def predict_faults(model, image_path, threshold=0.5):
    transform = T.Compose([T.ToTensor()])
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        predictions = model(img_tensor)[0]

    boxes = predictions['boxes']
    scores = predictions['scores']
    labels = predictions['labels']

    # Filter by confidence threshold
    filtered_boxes = []
    for box, score, label in zip(boxes, scores, labels):
        if score >= threshold:
            filtered_boxes.append((box, score.item(), label.item()))

    return img, filtered_boxes

# ---------- Draw Bounding Boxes ----------
def draw_boxes(image, boxes):
    draw = ImageDraw.Draw(image)
    for box, score, label in boxes:
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, y1), f"{score:.2f}", fill="red")
    return image

# ---------- Run on User Input ----------
# Example usage:
if __name__ == "__main__":
    model_path = "fault_detector_fasterrcnn.pth"
    test_image_path = "images/images.jpg"  # 👈 Replace with your image path

    model = load_model(model_path)
    image, boxes = predict_faults(model, test_image_path, threshold=0.5)
    result_image = draw_boxes(image, boxes)

    # Show the result
    result_image.show()

    # Optional: Save the result
    result_image.save(os.path.join("output","output.jpg"))
