import argparse
import os
import csv
import cv2
from ultralytics import YOLO

def main():
    # 1. Parse command line arguments exactly as requested
    parser = argparse.ArgumentParser(description="YOLOv8 Video Frame-by-Frame Inference Script")
    parser.add_argument("--video", type=str, required=True, help="Path to the input video file (mp4)")
    parser.add_argument("--weights", type=str, required=True, help="Path to the YOLOv8 model weights (.pt)")
    parser.add_argument("--output", type=str, required=True, help="Path to save the output CSV predictions file")
    args = parser.parse_args()

    # 2. Check if files exist
    if not os.path.exists(args.video):
        print(f"Error: Video file '{args.video}' not found.")
        return
    if not os.path.exists(args.weights):
        print(f"Error: Weights file '{args.weights}' not found.")
        return

    # 3. Load YOLOv8 Model
    model = YOLO(args.weights)
    
    # Class Mapping (YOLO model outputs 0, 1, 2 index. We map them to the expected output strings)
    class_map = {0: "standing", 1: "sitting", 2: "lying"}

    # 4. Open Video Stream
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{args.video}'.")
        return

    predictions = []
    frame_number = 0

    print("Processing video frames...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_number += 1
        
        # Run inference on single frame
        results = model(frame, verbose=False)
        
        best_cls_name = "unknown"
        best_conf = 0.0
        
        # Process detection results
        for r in results:
            boxes = r.boxes
            for box in boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                # Track the detection with the highest confidence score for the frame
                if conf > best_conf:
                    best_conf = conf
                    best_cls_name = class_map.get(cls, "unknown")
        
        # Store output records
        predictions.append({
            "frame_number": frame_number,
            "predicted_class": best_cls_name,
            "confidence_score": round(best_conf, 4) if best_cls_name != "unknown" else 0.0
        })

    cap.release()

    # 5. Write predictions to structured CSV file
    with open(args.output, mode='w', newline='') as csv_file:
        fieldnames = ['frame_number', 'predicted_class', 'confidence_score']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        
        writer.writeheader()
        for pred in predictions:
            writer.writerow(pred)
            
    print(f"Successfully processed {frame_number} frames. Predictions saved to: {args.output}")

if __name__ == "__main__":
    main()