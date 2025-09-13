import os
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np

# ----------- Custom Dataset Class -----------
class ClothFaultDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, transforms=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transforms = transforms
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        txt_path = os.path.join(self.annotation_dir, os.path.splitext(img_name)[0] + '.txt')

        # Load image
        img = Image.open(img_path).convert("RGB")
        width, height = img.size

        # Load annotations
        boxes = []
        labels = []
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                for line in f:
                    cls_id, x_c, y_c, w, h = map(float, line.strip().split())
                    x_c *= width
                    y_c *= height
                    w *= width
                    h *= height
                    x_min = x_c - w / 2
                    y_min = y_c - h / 2
                    x_max = x_c + w / 2
                    y_max = y_c + h / 2
                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(int(cls_id))  # Assuming all classes are '1'

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target


# ----------- Transforms -----------
transform = T.Compose([
    T.ToTensor()
])

# ----------- Dataset and DataLoader -----------
image_dir = 'train/images'
annotation_dir = 'train/labels'
dataset = ClothFaultDataset(image_dir, annotation_dir, transforms=transform)
data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

# ----------- Model Setup -----------
def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# Only background (0) and fault (1)
model = get_model(num_classes=2)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# ----------- Optimizer and Training Setup -----------
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# ----------- Training Loop -----------
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for images, targets in data_loader:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        epoch_loss += losses.item()

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    lr_scheduler.step()
    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f}")

print("Training complete.")

# ----------- Save Model -----------
torch.save(model.state_dict(), 'fault_detector_fasterrcnn.pth')
