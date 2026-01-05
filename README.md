# Airbnb Amenity Detection using YOLOv8

## Overview
This project is an end-to-end computer vision system that detects amenities in Airbnb property images.  
It uses a custom-trained YOLOv8 object detection model and a Flask-based web application for inference.

The goal is to automatically identify amenities such as beds, lamps, pillows, televisions, and furniture to help improve property listing accuracy and automation.

---

## Tech Stack
- Python
- YOLOv8 (Ultralytics)
- PyTorch
- OpenCV
- Flask
- HTML/CSS

---

## Dataset
- Custom dataset with **33 amenity classes**
- Images sourced and labeled using Roboflow
- Structured into train / validation / test splits
- Includes common Airbnb amenities such as:
  - Bed
  - Pillow
  - Table lamp
  - Television
  - Refrigerator
  - Sink
  - Microwave
  - Couch
  - Dresser

---

## Model Details
- Architecture: YOLOv8
- Image size: 768 × 768
- Training epochs: 15
- Final model file: airbnb_amenity_yolov8_final.pt


The model learns spatial and contextual features to detect multiple amenities within a single image.

---

## Application Features
- Upload an image of a property interior
- Run real-time amenity detection
- Display bounding boxes with confidence scores
- Supports multiple detections per image

---

## Project Structure

The model learns spatial and contextual features to detect multiple amenities within a single image.

---

## Application Features
- Upload an image of a property interior
- Run real-time amenity detection
- Display bounding boxes with confidence scores
- Supports multiple detections per image

---

## Project Structure
airbnb-deployment/
├── app.py
├── airbnb_amenity_yolov8_final.pt
├── detector/
├── utils/
├── templates/
├── uploads/
├── notebooks/
│ └── Airbnb_Amenity_Detection_YOLOv8.ipynb
└── requirements.txt


---

## How to Run Locally

```bash

1. Clone the repository

git clone https://github.com/steeltroops11/airbnb-amenity-detection-yolov8.git
cd airbnb-amenity-detection-yolov8
--------------------------------------------
2. Install dependencies

pip install -r requirements.txt
--------------------------------------------
3. Run The Flask App

python app.py
--------------------------------------------
4. Open in Browser

http://127.0.0.1:5000
--------------------------------------------
Results

The model successfully detects multiple amenities in real-world Airbnb-style images, including bedrooms and living spaces.

Example detections include:

i. Beds with high confidence

ii. Pillows on beds

iii. Table lamps and furniture

iv. Dressers and storage units

(Screenshots available in the repository.)
--------------------------------------------
Future Improvements:

i. Deploy using Docker

ii. Add cloud inference support

iii. Improve class balance and dataset size

iv. Add confidence-based filtering controls
--------------------------------------------
Author

Built by Navish as a hands-on computer vision and deployment project.
