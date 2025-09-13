from flask import Flask, request, jsonify, send_file, render_template_string
from ultralytics import YOLO
import torch
# import torch.nn as nn
# import cv2
import io
import os
import uuid
import shutil
from PIL import Image
# from torchvision import models, transforms
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
# import matplotlib.pyplot as plt
import torchvision


app = Flask(__name__)

# Load the YOLO model
yolo_model = YOLO("best.pt")

# HTML template for the UI
html_template = """
<!DOCTYPE html>
<html lang="en">
	<head>
		<meta charset="UTF-8" />
		<meta name="viewport" content="width=device-width, initial-scale=1.0" />
		<title>Fabric Defect Detection</title>
		<style>
			* {
				margin: 0;
				padding: 0;
				box-sizing: border-box;
				font-family: Arial, sans-serif;
			}

			/* Body Styling */
			body {
				display: flex;
				align-items: center;
				justify-content: center;
				min-height: 100vh;
				background: linear-gradient(135deg, #84fab0, #8fd3f4);
				padding: 20px;
			}

			/* Container Styling */
			.container {
				text-align: center;
				padding: 30px;
				background: white;
				border-radius: 10px;
				box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
				animation: fadeIn 1s ease-out;
				width: 100%;
				max-width: 400px;
			}

			h2 {
				font-size: 1.5rem;
				color: #333;
				margin-bottom: 20px;
				font-weight: 600;
			}

			/* Button Styling */
			.button {
				padding: 10px 20px;
				margin: 15px 5px;
				color: white;
				border: none;
				border-radius: 5px;
				cursor: pointer;
				font-size: 1rem;
				transition: transform 0.3s, background-color 0.3s;
				width: 100%;
				max-width: 120px;
			}

			.upload-btn {
				background-color: rgba(185, 175, 175, 0.911);
			}

			.submit-btn {
				background-color: #4169e1;
			}

			.reset-btn {
				background-color: #ff6347;
			}

			.button:hover {
				transform: scale(1.05);
			}

			.upload-btn:hover {
				background-color: grey;
			}

			.submit-btn:hover {
				background-color: #191970;
			}

			.reset-btn:hover {
				background-color: #dc143c;
			}

			/* Dropdown Styling */
			.dropdown {
				margin: 15px 0;
				padding: 10px;
				border: 1px solid #ccc;
				border-radius: 5px;
				font-size: 1rem;
				cursor: pointer;
				width: 100%;
				transition: border-color 0.3s ease;
			}

			.dropdown:focus {
				border-color: #4caf50;
			}

			/* Preview and Output Styling */
			.image-container {
				margin-top: 20px;
				max-width: 100%;
			}

			.image-container img {
				max-width: 100%;
				height: auto;
				border-radius: 8px;
				box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.2);
			}

			.image-box {
				display: flex;
			}

			.image-div {
				margin-right: 10px;
			}

			.image-label {
				font-size: 1rem;
				margin-top: 10px;
				color: #555;
				font-weight: 500;
			}

			/* Keyframes for animations */
			@keyframes fadeIn {
				from {
					opacity: 0;
					transform: translateY(-10px);
				}
				to {
					opacity: 1;
					transform: translateY(0);
				}
			}

			/* Responsive adjustments */
			@media (max-width: 600px) {
				h2 {
					font-size: 1.2rem;
				}

				.button,
				.dropdown {
					font-size: 0.9rem;
					padding: 8px 15px;
				}
			}
		</style>
	</head>
	<body>
		<div class="container">
			<h2>Fabric Defect Detection</h2>
			<input
				type="file"
				id="file-input"
				accept="image/*"
				onchange="previewImage(event)"
			/>
			<br /><br />
			<label for="model-select" >Select Model:</label>
			<select id="model-select" class="dropdown">
				<option value="yolo">YOLO</option>
				<option value="rcnn">RCNN</option>
			</select>
			<br /><br />
			<button class="button submit-btn" onclick="submitImage()">Submit</button>
			<div id="output" class="output-image" style="display: none">
				<h3 id="output-text"></h3>
				<img
					id="output-image"
					src=""
					alt="Output Image"
					style="max-width: 100%; height: auto; border-radius: 8px"
				/>
			</div>
		</div>

		<script>
			let selectedFile;

			function previewImage(event) {
				selectedFile = event.target.files[0];
			}

			async function submitImage() {
				if (!selectedFile) {
					alert("Please select an image to upload.");
					return;
				}

				const formData = new FormData();
				formData.append("file", selectedFile);
				formData.append("model", document.getElementById("model-select").value);

				try {
					const response = await fetch("/predict", {
						method: "POST",
						body: formData,
					});

					if (!response.ok) throw new Error("Image processing failed.");

					const data = await response.json();
					if (data.model === "yolo") {
						document.getElementById("output-image").src = data.image_url;
						document.getElementById("output-text").innerText =
							"Processed Image:";
						document.getElementById("output-image").style.display = "block";
					} else if (data.model === "rcnn") {
						document.getElementById("output-image").src = data.image_url;
						document.getElementById("output-text").innerText =
							"Processed Image:";
						document.getElementById("output-image").style.display = "block";
					}
					document.getElementById("output").style.display = "block";
				} catch (error) {
					console.error("Error:", error);
					alert("An error occurred while processing the image.");
				}
			}
		</script>
	</body>
</html>
"""


# ---------- Load Model ----------
def load_model(model_path, num_classes=2):
    # print("Model")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

# ---------- Inference on One Image ----------
def predict_faults(model, image_path, threshold=0.5):
    # print("Predicting")
    transform = T.Compose([T.ToTensor()])
    # print("after transform")
    img = Image.open(image_path).convert("RGB")
    # print("after img")
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

    # print("before with")
    with torch.no_grad():
        predictions = model(img_tensor)[0]
    # print("after with")
    boxes = predictions['boxes']
    scores = predictions['scores']
    labels = predictions['labels']

    # Filter by confidence threshold
    filtered_boxes = []
    # print("Startinng loop")
    for box, score, label in zip(boxes, scores, labels):
        if score >= threshold:
            filtered_boxes.append((box, score.item(), label.item()))
    # print("I am at end")
    return img, filtered_boxes

# ---------- Draw Bounding Boxes ----------
def draw_boxes(image, boxes):
    # print("drawind")
    draw = ImageDraw.Draw(image)
    for box, score, label in boxes:
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, y1), f"{score:.2f}", fill="red")
    return image


def preprocess_image_for_yolo(image):
    # Resize image to 640x640 for YOLO
    return image.resize((640, 640))


# Load RCNN model weights
rcnn_model = load_model("fault_detector_fasterrcnn.pth", num_classes=2)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "output"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    model_type = request.form.get("model", "yolo")

    try:
        unique_filename = str(uuid.uuid4()) + ".jpg"
        image_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        # image_path = unique_filename

        # Save the image from the request
        image = Image.open(io.BytesIO(file.read()))

        if model_type == "yolo":
            # Preprocess image for YOLO (resize to 640x640)
            image = preprocess_image_for_yolo(image)
            image.save(image_path)

            # Run YOLO model for prediction
            results = yolo_model.predict(image_path, save=True)
            # print(results)
            processed_image_path = os.path.join(results[0].save_dir, unique_filename)
            output_image_path = os.path.join(OUTPUT_FOLDER, unique_filename)
            # output_image_path = unique_filename
            shutil.move(str(processed_image_path), output_image_path)
            return jsonify({"model": "yolo", "image_url": "/output/" + unique_filename})
            # return jsonify({"model": "yolo", "image_url":  output_image_path})

        elif model_type == "rcnn":
            # Save the original image, preprocess and run RCNN model for defect detection
            image.save(image_path)
            model_path = "fault_detector_fasterrcnn.pth"
            # print("Hi")
            # test_image_path = "images.jpg"  # 👈 Replace with your image path

            model = load_model(model_path)
            Resimage, boxes = predict_faults(model, image_path, threshold=0.5)
            # print("I am back")
            result_image = draw_boxes(Resimage, boxes)
            # print("I am at the end")

            # Optional: Save the result
            # result_image.save("output_with_faults.jpg")
            result_image.save(os.path.join(OUTPUT_FOLDER,unique_filename))
            return jsonify({"model": "rcnn", "image_url": "/output/" + unique_filename})
            # return jsonify({"model": "rcnn", "image_url": "output_with_faults.jpg"})
            # return jsonify({"model": "rcnn", "image_url": "output_with_faults.jpg"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/output/<filename>")
def output_file(filename):
    # print("Hello")
    return send_file(os.path.join(OUTPUT_FOLDER, filename), mimetype="image/jpeg")


@app.route("/")
def home():
    return render_template_string(html_template)


if __name__ == "__main__":
    app.run(debug=True)
