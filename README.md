
# Florence2-YOLO Object Detection

An end-to-end object detection pipeline that uses Florence-2 for zero-shot detection and YOLOv8 for fine-tuning on custom classes: `person` and `pet` (`dog`, `cat`, `horse`).  
The project covers dataset creation, annotation filtering, augmentation strategies, training, evaluation, and visualization.

## Repository Structure

```
.
├── dataset_creator.ipynb         # Florence2-based dataset generation + COCO output
├── model_training.ipynb          # Training and evaluation of YOLOv8
├── inference_and_viz.ipynb       # Inference with preprocessing & test-time tools
├── object_detection_dataset/     # Saved images and COCO annotations
├── yolo_training/                # YOLOv8 outputs (runs, weights, plots)
├── documentation/                # TTA, Ensemble, DoF explanations
├── requirements.txt              # Python dependencies
└── README.md
```

## Project Pipeline

1. **Dataset Creation**
   - Collect images from Flickr30k
   - Run Florence-2 to extract `person`/`pet` annotations
   - Filter low-confidence detections and generate COCO dataset

2. **Conversion to YOLO**
   - Convert COCO annotations to YOLO format
   - Generate `dataset.yaml` for Ultralytics

3. **Training**
   - Baseline (no augmentation)
   - Full augmentation (mixup, mosaic, erasing, etc.)
   - AutoResume + checkpoint saving

4. **Evaluation**
   - mAP@0.5 and mAP@0.5:0.95 per class
   - Confusion matrix and PR curves

5. **Inference & Preprocessing**
   - CLAHE + sharpening + gamma correction
   - Annotated output images and batch statistics

## Results

### Confusion Matrix  
Insert here:  
`/yolo_training/custom_detection_augmented/confusion_matrix_normalized.jpg`

Example:  
`![Confusion Matrix](./yolo_training/custom_detection_augmented/confusion_matrix_normalized.jpg)`

### Sample Predictions  
Insert enhanced image vs. raw image comparison:  
- Original Image  
- Preprocessed Image  
- YOLOv8 Detection Results

## Key Observations

- High precision for `person`, slightly weaker recall for `pet`
- Misclassifications mostly happen between `pet` and background
- Florence-2 tends to over-label or under-label small objects like cats

## Work in Progress

| Feature                  | Status         |
|--------------------------|----------------|
| Model Enhancement        | YOLOv8-L used, upgradeable to YOLOv8-X |
| Class Balance            | Pet class needs more cat samples |
| Threshold Tuning         | Per-class score thresholds defined |
| Label Validation         | Manually check Florence-2 pet images |
| Dataset Balancing        | Ensure equal images per class and subclass |
| Hyperparameter Tuning    | Ongoing |

## How to Run

1. **Install requirements**

```bash
pip install -r requirements.txt
```

2. **Run Dataset Generator**

```bash
python dataset_creator.ipynb
```

3. **Train Model**

```bash
python model_training.ipynb
```

4. **Inference**

```bash
python inference_and_viz.ipynb
```

## License

MIT License © 2025 Vlad Pavlyuk  
Dataset built using Flickr30k (Creative Commons)

## TODOs

- [ ] Add more cat images to balance subclasses  
- [ ] Improve filtering logic for Florence detections  
- [ ] Enable Test-Time Augmentation (TTA) for evaluation  
- [ ] Document annotation QA process  

## Image Placeholders

- `results/label_examples.jpg`  
- `results/sample_inference.jpg`  
- `documentation/tta_explanation.txt`  
- `documentation/degrees_of_freedom_explanation.txt`  
- `runs/detect/val*/` evaluation snapshots  
