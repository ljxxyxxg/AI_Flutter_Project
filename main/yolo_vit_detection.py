from tensorflow.keras.models import load_model
from ultralytics import YOLO
from flask import Flask, request, jsonify
import numpy as np
import cv2
import os
import base64
import torch
from transformers import ViTForImageClassification, ViTFeatureExtractor
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# YOLOv8 모델
yolo_model = YOLO('best.pt')

# ViT 모델
model_directory = './model'
vit_model = ViTForImageClassification.from_pretrained(model_directory)
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')

# 클래스 
class_names = ['Banana', 'Broccoli', 'Cabbage', 'Carrot', 'Cucumber',
               'Garlic', 'Onion', 'Potato', 'Radish', 'Tomato']

@app.route('/predict', methods=['POST'])
def upload_file():
    vgt_list = []
    file = request.files['frame']
    img_bytes = file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # YOLO 모델로 객체 감지
    results = yolo_model(img)
    bounding_boxes = []
    for result in results[0].boxes:
        xmin, ymin, xmax, ymax = result.xyxy[0]
        confidence = result.conf[0]
        if confidence > 0.2:
            bounding_boxes.append((int(xmin), int(ymin), int(xmax), int(ymax)))
            cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
    
    output_image_path = os.path.join(app.config['OUTPUT_FOLDER'], 'output_with_bboxes.jpg')
    cv2.imwrite(output_image_path, img)

    # 바운딩 박스 영역 잘라내기 및 저장
    for i, (xmin, ymin, xmax, ymax) in enumerate(bounding_boxes):
        cropped_image = img[ymin:ymax, xmin:xmax]
        cropped_image_path = os.path.join(app.config['OUTPUT_FOLDER'], f'cropped_image_{i}.jpg')
        cv2.imwrite(cropped_image_path, cropped_image)

    # 잘라낸 이미지를 ViT로 이미지 분류
    for i in range(len(bounding_boxes)):
        img_path = os.path.join(app.config['OUTPUT_FOLDER'], f'cropped_image_{i}.jpg')
        predicted_class_index, predicted_class_name = classify_image(img_path)
        vgt_list.append(predicted_class_name)
        vgt_list = list(set(vgt_list))

    print("===============ViT 이미지 분류 결과 ==================" )
    print(vgt_list)
    
    _, buffer = cv2.imencode('.png', img)
    encoded_image = base64.b64encode(buffer).decode('utf-8')
    response = {
        'image': encoded_image,
        'classifications': vgt_list
    }
    return jsonify(response)


def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    inputs = feature_extractor(images=img, return_tensors="pt")
    return inputs


def classify_image(img_path):
    inputs = preprocess_image(img_path)
    outputs = vit_model(**inputs)
    logits = outputs.logits
    predicted_class_index = logits.argmax(-1).item()
    predicted_class_name = class_names[predicted_class_index]
    return predicted_class_index, predicted_class_name

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
