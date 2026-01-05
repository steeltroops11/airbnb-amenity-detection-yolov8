from ultralytics import YOLO
import cv2
import os
import numpy as np

class AmenityDetector:
    def __init__(self, model_path):
        # Convert to absolute path if relative (before checking existence)
        if not os.path.isabs(model_path):
            model_path = os.path.abspath(model_path)
        
        # Normalize the path (resolve any .. or . components)
        model_path = os.path.normpath(model_path)
        
        # Validate model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}. Current working directory: {os.getcwd()}")
        
        # Use absolute path string to ensure YOLO doesn't try to download
        self.model = YOLO(model_path)

    def detect(self, image_path, conf=0.25, draw_boxes=False):
        results = self.model(image_path, conf=conf)[0]

        detections = []
        boxes_data = []
        
        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = self.model.names[cls_id]
            confidence = float(box.conf[0])
            
            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            detections.append({
                "amenity": label,
                "confidence": round(confidence, 3),
                "bbox": {
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2)
                }
            })
            
            boxes_data.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "label": label,
                "confidence": confidence
            })

        annotated_image_path = None
        if draw_boxes:
            annotated_image_path = self._draw_boxes(image_path, boxes_data)
        
        return detections, annotated_image_path
    
    def _draw_boxes(self, image_path, boxes_data):
        """Draw bounding boxes on the image"""
        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        # Color palette for different amenities (BGR format for OpenCV)
        colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
        ]
        
        # Draw each bounding box
        for idx, box_info in enumerate(boxes_data):
            x1, y1, x2, y2 = box_info["bbox"]
            label = box_info["label"]
            confidence = box_info["confidence"]
            
            # Select color based on index (cycling through colors)
            color = colors[idx % len(colors)]
            
            # Draw rectangle with thicker lines for better visibility
            thickness = 3
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            
            # Prepare label text
            label_text = f"{label} {confidence:.2f}"
            
            # Get text size for background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            font_thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, font, font_scale, font_thickness
            )
            
            # Draw label background with padding
            padding = 5
            cv2.rectangle(
                img,
                (x1, y1 - text_height - baseline - padding * 2),
                (x1 + text_width + padding * 2, y1),
                color,
                -1
            )
            
            # Draw label text in white for better contrast
            cv2.putText(
                img,
                label_text,
                (x1 + padding, y1 - text_height - padding),
                font,
                font_scale,
                (255, 255, 255),  # White text
                font_thickness
            )
        
        # Save annotated image
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        annotated_path = os.path.join(
            os.path.dirname(image_path),
            f"{base_name}_annotated.jpg"
        )
        cv2.imwrite(annotated_path, img)
        
        return annotated_path
