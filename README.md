# Computer Vision For Surgical Application

This project implements a computer vision system for surgical tool detection and pose estimation using synthetic data generation and deep learning approaches.

## Project Overview

This system provides a complete pipeline for:
1. Synthetic surgical tool dataset generation
2. Training YOLOv11-based pose estimation
3. Real-time prediction and visualization
4. Data mixing and processing utilities

## Prerequisites

- Python 3.8+
- Blender 3.6+
- CUDA-capable GPU (recommended for training)
- Required Python packages (see `requirements.txt`)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Nika0990/Computer_Vision_For_Surgical_Application.git
cd Computer_Vision_For_Surgical_Application
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Install Blender (if not already installed):
```bash
# For Ubuntu/Debian
sudo apt-get install blender
```

## Project Structure

- `synthetic_data_generator.py`: Generate synthetic dataset using CAD models and HDRI lighting
- `instrument_pose_yolov11.py`: YOLOv11-based instrument pose estimation implementation
- `predict.py`: Run inference on images/video
- `video.py`: Process video files
- `visualize_annotation.py`: Visualize dataset annotations
- `mix_and_resplit_yolo_pose.py`: Data mixing and splitting utilities
- `yolo_train_with_pred.py`: YOLO training with predictions
- `video_to_yolo_pose.py`: Convert video data to YOLO pose format

### Data Directories
- `ground_truth_sample/`: Sample ground truth data
  - `ann/`: JSON annotations
  - `images/`: Source images
  - `viz/`: Visualization overlays

## Detailed Usage Guide

### 1. Synthetic Data Generation (`synthetic_data_generator.py`)

```bash
blender --background --factory-startup --python synthetic_data_generator.py \
    --models_dir /path/to/cad/models \
    --hdris_dir /path/to/hdri/maps \
    --camera /path/to/camera.json \
    --keypoints /path/to/keypoints.json \
    --output_dir ./synthetic_dataset \
    --num_images 3000 \
    --render_samples 64
```

Parameters:
- Required:
  - `--models_dir`: Directory containing CAD models (.obj files)
  - `--hdris_dir`: Directory of HDR environment maps for lighting
  - `--camera`: Path to camera configuration JSON file
  - `--keypoints`: Path to keypoints mapping JSON file
- Optional:
  - `--output_dir`: Output directory (default: ./synthetic_dataset)
  - `--num_images`: Number of images to generate (default: 1000)
  - `--seed`: Random seed for reproducibility
  - `--render_samples`: Number of render samples (default: 32)
  - `--fast_mode`: Enable fast rendering mode
  - `--lens_cap_mm`: Lens focal length in mm (default: 24.0)
  - `--hdri_zoom_min/max`: HDRI zoom range (default: 800-1600)
  - `--two_tools_prob`: Probability of generating two tools (default: 0.40)

Size and Probability Settings:
- Tool size ranges (small/medium/large/close) with their probabilities
- Controlled by `--small_lo/hi`, `--med_lo/hi`, `--large_lo/hi`, `--close_lo/hi`
- Probability distribution: `--prob_small`, `--prob_medium`, `--prob_large`, `--prob_close`

### 2. YOLOv11 Training (`instrument_pose_yolov11.py`)

This script handles both dataset preparation and model training in a two-step process.

#### Step 1: Prepare Dataset
```bash
python instrument_pose_yolov11.py \
  --step prepare \
  --root /path/to/dataset \
  --out_dir /path/to/output \
  --aug_copies 1 \
  --tweezer_boost 0 \
  --band_prob 0.35 \
  --glove_prob 0.55 \
  --grip_prob 0.45 \
  --band_per_image 0.55 \
  --glove_per_image 0.55 \
  --grip_per_image 0.55 \
  --or_prob 0.50 \
  --video_prob 0.35
```

Preparation Parameters:
- Required:
  - `--step`: Operation mode ('prepare', 'train', or 'prepare+train')
  - `--root`: Dataset root directory containing images/ and ann/ folders
  - `--out_dir`: Output directory for processed dataset
- Dataset Split:
  - `--val_pct`: Validation set percentage (default: 0.1)
  - `--seed`: Random seed for reproducibility (default: 42)
- Augmentation Control:
  - `--aug_copies`: Number of augmented copies per image (default: 1)
  - `--tweezer_boost`: Extra copies only for images with tweezers (default: 0)
- Occlusion Probabilities:
  - `--band_prob`: Probability of band occlusions (default: 0.35)
  - `--glove_prob`: Probability of glove occlusions (default: 0.55)
  - `--grip_prob`: Probability of hand-grip occluder (default: 0.40)
  - `--or_prob`: Probability of OR lighting effects (default: 0.50)
  - `--video_prob`: Probability of video-like effects (default: 0.35)
- Occlusion Intensity:
  - `--band_per_image`: Band occlusion strength per image (0-1, default: 0.60)
  - `--glove_per_image`: Glove patch intensity per image (0-1, default: 0.60)
  - `--grip_per_image`: Hand-grip occluder intensity (0-1, default: 0.60)

#### Step 2: Train Model
```bash
python instrument_pose_yolov11.py \
  --step train \
  --out_dir /path/to/prepared_data \
  --model yolo11l-pose.pt \
  --epochs 100 \
  --imgsz 832 \
  --batch 8 \
  --device 0 \
  --run_name train_hdri_background
```

Training Parameters:
- Required:
  - `--out_dir`: Directory containing prepared dataset
  - `--model`: Initial model weights (default: yolo11l-pose.pt)
- Training Configuration:
  - `--epochs`: Number of training epochs (default: 100)
  - `--imgsz`: Input image size (default: 1024)
  - `--batch`: Batch size (-1 for auto-selection)
  - `--device`: CUDA device index or 'cpu' (default: '0')
  - `--workers`: Number of worker threads (default: auto)
- Output Control:
  - `--run_project`: Project output directory (default: runs/pose)
  - `--run_name`: Name of this training run (default: train_hdri_background)

### 3. Video Processing and Conversion (`video_to_yolo_pose.py`)

```bash
python video_to_yolo_pose.py \
    --weights /path/to/model.pt \
    --source /path/to/video.mp4 \
    --out_dir /path/to/output/dir \
    --imgsz 1280 \
    --conf 0.7 \
    --kpt_conf 0.7 \
    --iou 0.5 \
    --device 0 \
    --stride_frames 1
```

Parameters:
- Required:
  - `--weights`: Path to YOLO pose model weights (.pt file)
  - `--source`: Input video path or webcam index
  - `--out_dir`: Output directory for images/ and labels/
- Model Configuration:
  - `--imgsz`: Input image size (default: 1280)
  - `--conf`: Confidence threshold (default: 0.25)
  - `--iou`: NMS IoU threshold (default: 0.5)
  - `--device`: CUDA device index or 'cpu' (default: '0')
  - `--max_det`: Maximum detections per frame (default: 50)
- Processing Options:
  - `--stride_frames`: Process every Nth frame (default: 1)
  - `--kpt_conf`: Minimum keypoint confidence (default: 0.20)
  - `--save_empty`: Save frames with no detections

### 4. Visualization Tool (`visualize_annotation.py`)

```bash
python visualize_annotation.py \
    --ann_file /path/to/annotation.json \
    --no_show
```

Parameters:
- `--ann_file`: Path to annotation JSON file
- `--no_show`: Skip displaying visualization window (optional)

### 5. Data Mixing and Splitting (`mix_and_resplit_yolo_pose.py`)

```bash
python mix_and_resplit_yolo_pose.py 
    --base_yolo /path/to/base/dataset 
    --extra /path/to/extra1 /path/to/extra2 
    --out_dir /path/to/output 
    --val_pct 0.10 
    --seed 42 
    --force
```

Parameters:
- Required:
  - `--base_yolo`: Base YOLO dataset directory
  - `--extra`: One or more extra folders to mix in
  - `--out_dir`: Output directory for mixed dataset
- Options:
  - `--val_pct`: Validation set percentage (default: 0.10)
  - `--seed`: Random seed (default: 42)
  - `--force`: Clear output directory if it exists

### 6. YOLO Training with Predictions (`yolo_train_with_pred.py`)

This script trains YOLOv11-pose directly from an existing YOLO dataset. It expects a standard YOLO dataset structure:
```
yolo_mixed/
  ├─ train/
  │   ├─ images/*.jpg|png
  │   └─ labels/*.txt
  ├─ val/
  │   ├─ images/*.jpg|png
  │   └─ labels/*.txt
  ├─ data.yaml
  └─ manifest.csv   (optional)
```

```bash
python yolo_train_with_pred.py 
    --data /path/to/data.yaml 
    --model yolo11l-pose.pt 
    --epochs 100 
    --imgsz 832 
    --batch 8 
    --device 0 
    --run_project runs/pose 
    --run_name train_from_yolo_mixed 
```

Parameters:
- Required:
  - `--data`: Path to data.yaml or dataset folder containing data.yaml
- Model Configuration:
  - `--model`: Initial model weights (default: yolo11l-pose.pt)
  - `--epochs`: Number of training epochs (default: 100)
  - `--imgsz`: Input image size (default: 1024)
  - `--batch`: Batch size (-1 for auto) (default: -1)
  - `--device`: CUDA device or 'cpu' (default: '0')
  - `--workers`: Number of worker threads (auto if not specified)
- Run Configuration:
  - `--run_project`: Project output directory (default: runs/pose)
  - `--run_name`: Name of this training run (default: train_from_yolo_mixed)

The script automatically patches data.yaml if needed with:
- Single class: ["instrument"]
- 5 keypoints
- Appropriate flip indices for left/right consistency

### 7. Model Inference (`predict.py`)

```bash
python predict.py \
    --weights /path/to/weights.pt \
    --source /path/to/image \
    --output annotated.png \
    --conf 0.7 \
    --iou 0.5 \
    --imgsz 1280 \
    --device 0 \
    --kpt_conf 0.5 \
    --thickness 2 \
    --radius 6 \
    --labels 
    --output /path/to/your_output.jpg
```

Parameters:
- Required:
  - `--weights`: Path to trained pose model weights (.pt file)
  - `--source`: Input image path
- Output:
  - `--output`: Output image path (default: annotated.png)
- Model Configuration:
  - `--conf`: Box confidence threshold (default: 0.4)
  - `--iou`: NMS IoU threshold (default: 0.5)
  - `--imgsz`: Inference size, multiple of 32 (default: 1280)
  - `--device`: CUDA device or 'cpu' (default: '0')
  - `--kpt_conf`: Minimum keypoint confidence to draw (default: 0.20)

### 7. Video Processing (`video.py`)

```bash
python video.py \
    --weights /path/to/weights.pt \
    --source /path/to/video.mp4 \
    --output annotated.mp4 \
    --conf 0.7 \
    --kpt_conf 0.7 \
    --iou 0.5 \
    --imgsz 1280 \
    --device 0 \
    --stride_frames 1 \
```

Parameters:
- Required:
  - `--weights`: Path to trained pose model weights (.pt file)
  - `--source`: Input video path or webcam index
  - `--output`: Output video path (default: annotated.mp4)
- Model Configuration:
  - `--conf`: Box confidence threshold (default: 0.4)
  - `--iou`: NMS IoU threshold (default: 0.5)
  - `--imgsz`: Inference size, multiple of 32 (default: 1280)
  - `--device`: CUDA device or 'cpu' (default: '0')
- Processing Options:
  - `--stride_frames`: Process every Nth frame (default: 1)
  - `--kpt_conf`: Minimum keypoint confidence to draw

## Data Format

### Annotation Format
The project uses a JSON annotation format:
```json
{
    "images": [
        {
            "id": int,
            "file_name": str,
            "width": int,
            "height": int
        }
    ],
    "annotations": [
        {
            "id": int,
            "image_id": int,
            "category_id": int,
            "keypoints": [x1, y1, v1, x2, y2, v2],
            "num_keypoints": int
        }
    ],
    "categories": [
        {
            "id": int,
            "name": str
        }
    ]
}
```

## Model Architecture and Training Phases


### Training Phases Overview

The training process is divided into multiple phases to achieve optimal performance. Pre-trained weights for both phases are included in this repository:

#### Phase 2 Weights (`weights_phase2.pt`)
- Initial training phase using synthetic data
- Focuses on learning basic tool detection and keypoint estimation
- Uses purely synthetic images with controlled lighting and backgrounds
- Helps model learn fundamental tool features and geometric relationships

Due to file size limitations, the pre-trained weights are hosted using Git LFS (Large File Storage):
- Phase 2 weights: [Download weights_phase2.pt](https://github.com/Nika0990/Computer_Vision_For_Surgical_Application/raw/main/weights_phase2.pt)

#### Phase 3 Weights (`weights_phase3.pt`)
- Advanced training phase incorporating real surgical data
- Fine-tunes the model on actual surgical scenarios
- Improves robustness to real-world variations
- Enhances performance on actual surgical videos
- Addresses domain adaptation between synthetic and real data

Download link for final weights:
- Phase 3 weights: [Download weights_phase3.pt](https://github.com/Nika0990/Computer_Vision_For_Surgical_Application/raw/main/weights_phase3.pt)

Note: If you're cloning the repository, make sure to install Git LFS first:
```bash
git lfs install
git clone https://github.com/Nika0990/Computer_Vision_For_Surgical_Application.git
cd Computer_Vision_For_Surgical_Application
git lfs pull
```

### Detailed File Descriptions

1. `synthetic_data_generator.py`
   - Generates synthetic training data using Blender
   - Creates realistic tool renderings with HDRI lighting
   - Supports multiple tool sizes and poses
   - Generates accurate keypoint annotations
   - Features:
     * Dynamic lighting conditions
     * Random tool positioning and rotation
     * Automatic size adaptation
     * Blood stain effects (optional)
     * Multi-tool scene generation

2. `instrument_pose_yolov11.py`
   - Implements the YOLOv11-based pose estimation model
   - Handles both training and validation
   - Supports multi-GPU training
   - Includes custom loss functions for keypoint detection
   - Implements data augmentation strategies

3. `mix_and_resplit_yolo_pose.py`
   - Combines synthetic and real datasets
   - Maintains proper train/validation splits
   - Handles data balancing between synthetic and real samples
   - Ensures consistent annotation formats

4. `video_to_yolo_pose.py`
   - Converts video data to YOLO training format
   - Extracts frames at specified intervals
   - Generates appropriate directory structure
   - Preserves frame sequence information

5. `yolo_train_with_pred.py`
   - Specialized training script with prediction integration
   - Supports continued training from checkpoints
   - Implements adaptive learning rate strategies
   - Includes validation metrics calculation

6. `predict.py`
   - Handles real-time inference
   - Supports both image and video input
   - Implements efficient post-processing
   - Features visualization options
   - Supports batch processing

7. `video.py`
   - Processes video streams
   - Implements frame buffering for smooth processing
   - Handles various video formats
   - Supports real-time visualization

8. `visualize_annotation.py`
   - Debug and visualization tool
   - Renders keypoint annotations
   - Supports both synthetic and real data
   - Helps in data quality assessment

## Results Visualization

Visualization tools are provided to overlay predictions on images/videos:
- `visualize_annotation.py`: For dataset visualization
- Built-in visualization in `predict.py`
- Real-time visualization capabilities in video processing
