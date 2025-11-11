# License_Plate_Detection_with_YOLOV8n_and_PaddleOCR
Indian License Plate Detection &amp; OCR System - Real-time detection with 90% accuracy using YOLO and PaddleOCR
License Plate Detection and OCR System
Project Documentation

1. INTRODUCTION

1.1 Project Overview
This project implements an automated License Plate Recognition (LPR) system using state-of-the-art deep learning technologies. The system combines YOLOv8 for license plate detection with PaddleOCR for text extraction, creating a complete end-to-end solution for Indian vehicle license plate recognition.

1.2 Project Objectives
• Accurately detect license plates from vehicle images across various Indian states
• Extract alphanumeric text from detected plates using OCR technology
• Handle diverse conditions including different lighting, angles, and plate formats
• Achieve high accuracy through image preprocessing and enhancement techniques
• Provide real-time processing capabilities for practical deployment

1.3 Technology Stack
• Deep Learning Framework: YOLOv8n (Ultralytics)
• OCR Engine: PaddleOCR 3.3.1
• Image Processing: OpenCV 4.10
• Programming Language: Python 3.9.25
• Development Environment: Jupyter Notebook
• Additional Libraries: PyTorch, NumPy, Pandas, Matplotlib


2. DATASET

2.1 Dataset Overview
The dataset consists of Indian vehicle license plate images collected from multiple sources, representing diverse real-world scenarios across different states.

2.2 Dataset Composition
• Total Images: Multiple thousand images from 35 Indian states
• Data Sources:
  - State-wise OLX vehicle listings
  - Google Images collection
  - Video frame extractions
• Annotation Format: XML files with bounding box coordinates
• Image Formats: JPG, JPEG, PNG


2.4 Data Preprocessing
• XML to YOLO format conversion (normalized coordinates)
• Train-validation split: 80-20 ratio
• Image resolution: Variable (handled by model)
• Data augmentation: Applied during training by YOLOv8

2.5 Label Format
Each label file contains:
- Class ID: 0 (license_plate)
- Bounding box: x_center, y_center, width, height (normalized)
Example: 0 0.512 0.643 0.185 0.092


3. MODELS AND ARCHITECTURE

3.1 YOLOv8n Model

3.1.1 Model Selection
YOLOv8n (nano) was chosen for its optimal balance between:
• Speed: Real-time inference capability
• Accuracy: High detection precision
• Size: Lightweight model (6.2M parameters)
• Efficiency: Suitable for deployment on various devices

3.1.2 Model Architecture
• Backbone: CSPDarknet with C2f modules
• Neck: PANet (Path Aggregation Network)
• Head: Decoupled detection head
• Input Size: 416x416 pixels
• Output: Bounding boxes with confidence scores

3.1.3 Training Configuration
• Epochs: 50 (with early stopping patience=5)
• Batch Size: 4
• Image Size: 416x416
• Optimizer: SGD with momentum
• Learning Rate: Auto-adjusted
• Device: CPU (for compatibility)
• Confidence Threshold: 0.25


3.2 PaddleOCR Model

3.2.1 OCR Engine Overview
PaddleOCR is an industrial-grade OCR system developed by PaddlePaddle, offering:
• Multi-language support (configured for English)
• High accuracy for printed text
• Robust performance on rotated/tilted text
• Lightweight and efficient

3.2.2 OCR Configuration
• Model Version: PaddleOCR 3.3.1
• Language: English (optimized for alphanumeric)
• Text Orientation: Enabled (use_textline_orientation=True)
• Backend: PaddlePaddle 3.2.1

3.2.3 OCR Pipeline
The OCR system operates in three stages:
1. Text Detection: Locates text regions in the image
2. Text Recognition: Converts detected regions to text
3. Post-processing: Formats and validates output

3.3 Integration Architecture

The complete system workflow:

1. Input Image → 2. YOLO Detection → 3. Plate Extraction
    ↓
4. Image Preprocessing:
   • Resize (minimum height: 100px)
   • Grayscale conversion
   • Denoising (fastNlMeans)
   • Contrast enhancement (CLAHE)
   • Adaptive thresholding
    ↓
5. Dual OCR Processing:
   • OCR on original plate
   • OCR on enhanced plate
   • Select best result (highest confidence)
    ↓
6. Output: License plate text + confidence scores
    ↓
7. CSV Export: Save results with timestamp


4. IMAGE PREPROCESSING TECHNIQUES

4.1 Preprocessing Pipeline
To maximize OCR accuracy, multiple image enhancement techniques are applied:

4.1.1 Resizing
• Purpose: Ensure sufficient resolution for OCR
• Method: Scale plate height to minimum 100 pixels
• Interpolation: Cubic interpolation for quality

4.1.2 Grayscale Conversion
• Purpose: Reduce complexity, focus on text structure
• Method: OpenCV BGR to Gray conversion

4.1.3 Denoising
• Technique: Non-local Means Denoising
• Parameters: h=10, templateWindowSize=7, searchWindowSize=21
• Effect: Removes noise while preserving edges

4.1.4 Contrast Enhancement (CLAHE)
• Technique: Contrast Limited Adaptive Histogram Equalization
• Parameters: clipLimit=3.0, tileGridSize=(8,8)
• Effect: Enhances local contrast, improves text visibility

4.1.5 Adaptive Thresholding
• Method: Gaussian adaptive thresholding
• Purpose: Binarize image for better text recognition
• Effect: Creates clear text boundaries


4.2 Dual OCR Strategy
The system runs OCR twice:
1. Original plate image → Text1 + Score1
2. Enhanced/preprocessed plate → Text2 + Score2
3. Select result with higher confidence score

This approach ensures optimal accuracy across varying image qualities.


5. IMPLEMENTATION AND RESULTS

5.1 Notebook Structure

5.1.1 number_plate_detection.ipynb (Training Notebook)
• Cell 1: Install dependencies
• Cell 2-3: Import libraries and setup
• Cell 4: Create dataset directories
• Cell 5: XML to YOLO conversion function
• Cell 6-7: Prepare dataset (split and convert)
• Cell 8: Create configuration file (dataset.yaml)
• Cell 9: Train YOLOv8 model
• Cell 10-11: Test trained model
• Cell 12-13: Visualize training metrics

5.1.2 license_plate_ocr.ipynb (Detection & OCR Notebook)
• Cell 1-2: Import libraries and load models
• Cell 3: Load and display test image
• Cell 4: Detect license plates with YOLO
• Cell 5: Extract plate region
• Cell 6: Preprocess plate with visualizations
• Cell 7: OCR on original plate
• Cell 8: OCR on enhanced plate
• Cell 9: Visualize final result with annotations
• Cell 10: Save results to CSV
• Cell 11-12: Batch test multiple state images

5.2 System Performance

5.2.1 Detection Performance
• Detection Confidence: Average 85-90%
• Detection Speed: Real-time (suitable for video)
• False Positive Rate: Low (< 5%)
• Robustness: Handles various angles and lighting

5.2.2 OCR Performance
• OCR Confidence: Average 87-93%
• Character Accuracy: High for clear images
• Format Support: All Indian license plate formats
• State Coverage: Tested on 35+ states

5.2.3 Sample Results
Test cases demonstrated:
• MH01AV8868: 88.3% detection, 93.2% OCR
• PY01AP0555: 91.3% detection, high accuracy
• DL plates: Successfully recognized
• KA plates: Successfully recognized

5.3 Output Format

5.3.1 CSV Export Structure
Results are saved with the following fields:
• timestamp: Date and time of detection
• image_name: Source image filename
• license_plate: Extracted text
• detection_confidence: YOLO confidence score
• ocr_confidence: OCR confidence score
• bbox: Bounding box coordinates (x1,y1,x2,y2)

5.3.2 Visual Output
• Annotated images with green bounding boxes
• License plate text displayed above each detection
• Confidence scores shown in parentheses
• Grid visualization for batch processing


6. APPLICATIONS AND FUTURE WORK

6.1 Practical Applications
• Automated Parking Management Systems
• Traffic Violation Detection
• Toll Collection Systems
• Vehicle Access Control
• Security and Surveillance Systems

6.3 Future Enhancements
Implement an automated license plate detection system to record vehicle entry and exit times, enabling calculation of parking duration.

6.4 Challenges Addressed
• Variable lighting conditions: Preprocessing handles this
• Different plate formats: Model trained on diverse dataset
• Tilted/angled plates: Text orientation detection enabled
• Low resolution images: Resizing and enhancement applied
• Noise and blur: Denoising techniques implemented




7. CONCLUSION

This project successfully demonstrates a complete License Plate Recognition system that combines modern deep learning techniques with traditional image processing methods. The system achieves high accuracy in detecting and recognizing Indian vehicle license plates across 35+ states, handling various real-world challenges including lighting variations, angles, and plate conditions.

The dual OCR strategy with preprocessing significantly improves recognition accuracy, while the comprehensive logging system ensures traceability and facilitates system monitoring. The modular architecture allows for easy maintenance, updates, and integration with larger systems.

Key achievements:
 Successful YOLOv8 model training with optimized parameters
 Integration of PaddleOCR for robust text extraction
 Implementation of preprocessing techniques
 Comprehensive testing across multiple Indian states
 Automated CSV export for data management
 Visualization tools for performance monitoring



TECHNICAL SPECIFICATIONS

Software Requirements:
• Python 3.9.25
• YOLOv8 (Ultralytics 8.3.222)
• PaddleOCR 3.3.1
• PaddlePaddle 3.2.1
• OpenCV 4.10.0.84
• PyTorch 2.7.1
• Pandas, NumPy, Matplotlib

Model Files:
• Pre-trained: yolov8n.pt
• Trained model: runs/detect/train5/weights/best.pt
• PaddleOCR models: Auto-downloaded on first run

REFERENCES

1. Ultralytics YOLOv8 Documentation
   https://docs.ultralytics.com/

2. PaddleOCR Documentation
   https://github.com/PaddlePaddle/PaddleOCR

3. OpenCV Documentation
   https://docs.opencv.org/
