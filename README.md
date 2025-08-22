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
python synthetic_data_generator.py \
    --models_dir /path/to/cad/models \
    --hdris_dir /path/to/hdri/maps \
    --camera /path/to/camera.json \
    --keypoints /path/to/keypoints.json \
    --output_dir ./synthetic_dataset \
    --num_images 1000 \
    --render_samples 32
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

```bash
python instrument_pose_yolov11.py \
    --data /path/to/data.yaml \
    --cfg /path/to/model/config.yaml \
    --weights /path/to/weights.pt \
    --epochs 100 \
    --batch-size 16 \
    --img-size 640 \
    --device 0
```

Parameters:
- `--data`: Path to data configuration file
- `--cfg`: Path to model configuration file
- `--weights`: Path to initial weights (optional)
- `--epochs`: Number of training epochs
- `--batch-size`: Training batch size
- `--img-size`: Input image size
- `--device`: CUDA device (0, 1, cpu)

### 3. Video Processing and Conversion (`video_to_yolo_pose.py`)

```bash
python video_to_yolo_pose.py \
    --input /path/to/video.mp4 \
    --output /path/to/output/dir \
    --frame-step 1 \
    --start-frame 0 \
    --end-frame -1
```

Parameters:
- `--input`: Input video file path
- `--output`: Output directory for extracted frames
- `--frame-step`: Process every nth frame (default: 1)
- `--start-frame`: Starting frame number (default: 0)
- `--end-frame`: Ending frame number (-1 for all frames)

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
python mix_and_resplit_yolo_pose.py \
    --synthetic /path/to/synthetic/data \
    --real /path/to/real/data \
    --output /path/to/output \
    --split-ratio 0.8
```

Parameters:
- `--synthetic`: Path to synthetic dataset
- `--real`: Path to real dataset
- `--output`: Output directory for mixed dataset
- `--split-ratio`: Train/val split ratio

### 6. Model Inference (`predict.py`)

```bash
python predict.py \
    --weights /path/to/weights.pt \
    --source /path/to/image_or_video \
    --img-size 640 \
    --conf-thres 0.25 \
    --iou-thres 0.45 \
    --device 0
```

Parameters:
- `--weights`: Path to model weights
- `--source`: Path to input image/video
- `--img-size`: Input image size
- `--conf-thres`: Confidence threshold
- `--iou-thres`: NMS IoU threshold
- `--device`: CUDA device (0, 1, cpu)

### 7. Video Processing (`video.py`)

```bash
python video.py \
    --input /path/to/input.mp4 \
    --output /path/to/output.mp4 \
    --model /path/to/weights.pt \
    --conf-thres 0.25 \
    --fps 30
```

Parameters:
- `--input`: Input video file
- `--output`: Output video file
- `--model`: Path to model weights
- `--conf-thres`: Confidence threshold
- `--fps`: Output video FPS

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

The system uses YOLOv11 for object detection and pose estimation, with custom modifications for surgical tool tracking. The model is designed to detect tool endpoints and estimate their 3D pose in real-time.

### Training Phases Overview

The training process is divided into multiple phases to achieve optimal performance. Pre-trained weights for both phases are included in this repository:

#### Phase 2 Weights (`weights_phase2.pt`)
- Initial training phase using synthetic data
- Focuses on learning basic tool detection and keypoint estimation
- Uses purely synthetic images with controlled lighting and backgrounds
- Helps model learn fundamental tool features and geometric relationships

Due to file size limitations, the pre-trained weights are hosted using Git LFS (Large File Storage):
- Phase 2 weights: [Download weights_phase2.pt](https://github.com/Nika0990/Computer_Vision_For_Surgical_Application/raw/master/weights_phase2.pt)

#### Phase 3 Weights (`weights_phase3.pt`)
- Advanced training phase incorporating real surgical data
- Fine-tunes the model on actual surgical scenarios
- Improves robustness to real-world variations
- Enhances performance on actual surgical videos
- Addresses domain adaptation between synthetic and real data

Download link for final weights:
- Phase 3 weights: [Download weights_phase3.pt](https://github.com/Nika0990/Computer_Vision_For_Surgical_Application/raw/master/weights_phase3.pt)

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
