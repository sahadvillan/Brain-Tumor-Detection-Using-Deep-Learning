# Brain Tumor Detection Using Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Hugging Face Models](https://img.shields.io/badge/🤗%20Models-Available-yellow)](https://huggingface.co/Heet010/brain-tumor-detection-models)

An advanced AI system for brain tumor detection using **ResNet for classification** and **ResUNet for segmentation** of MRI scans. This project implements state-of-the-art deep learning techniques with comprehensive training, evaluation, and visualization capabilities.

## 🧠 Overview

This project combines two powerful deep learning approaches:
- **ResNet (Residual Neural Network)** for binary brain tumor classification
- **ResUNet (Residual U-Net)** for precise pixel-wise tumor segmentation
- **Memory-optimized training** with mixed precision for efficient GPU utilization
- **Comprehensive evaluation** with detailed metrics and visualizations

## ✨ Key Features

- **Dual Architecture**: Classification + Segmentation in one pipeline
- **Advanced Data Preprocessing**: Albumentations-based augmentation pipeline
- **Memory Optimization**: Mixed precision training with gradient scaling
- **Comprehensive Metrics**: IoU, Dice coefficient, accuracy, F1-score, and more
- **TensorBoard Integration**: Real-time training visualization
- **Robust Evaluation**: Detailed model performance analysis
- **Modular Design**: Clean, well-documented code structure

## 🏗️ Model Architecture

### Classification Model (ResNet)
- **Base**: ResNet-50 with pretrained ImageNet weights
- **Custom Head**: Dropout + Dense layers for regularization
- **Loss Function**: Focal Loss for class imbalance handling
- **Output**: Binary classification (Tumor/No Tumor)

### Segmentation Model (ResUNet)
- **Architecture**: U-Net with residual connections
- **Encoder**: ResNet-style blocks with skip connections
- **Decoder**: Upsampling with feature concatenation
- **Loss Function**: Combined CrossEntropy + Dice Loss
- **Output**: Pixel-wise segmentation masks

## 📊 Dataset

Uses the **LGG MRI Segmentation Dataset** from Kaggle:
- **Source**: [Brain MRI Segmentation](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)
- **Content**: Lower Grade Glioma MRI scans with masks
- **Format**: TIFF images with corresponding segmentation masks
- **Structure**: Patient-wise folders with multiple slices per patient

## 🚀 Quick Start

### Prerequisites

Install the required dependencies:

```bash
pip install -r requirements.txt
```

**Key Dependencies:**
```
torch==2.7.1+cu118
torchvision==0.22.1+cu118
albumentations==2.0.8
opencv-python==4.12.0.88
matplotlib==3.10.5
scikit-learn==1.7.1
tensorboard==2.20.0
huggingface-hub==0.34.4
```

### Installation

1. **Clone the repository**
   ```bash
   [git clone https://github.com/sahadvillan/Brain-Tumor-Detection-Using-Deep-Learning.git]
   cd Brain-Tumor-Detection-Using-Deep-Learning
   ```

2. **Download the dataset**
   ```bash
   # Download from Kaggle
   kaggle datasets download -d mateuszbuda/lgg-mri-segmentation
   unzip lgg-mri-segmentation.zip -d data/
   
   ```

### Usage

#### Option 1: Complete Pipeline (Recommended)

```bash
# Run the complete pipeline (preprocessing + training + evaluation)
python main.py --mode full --task both

# Run only training
python main.py --mode train --task classification
python main.py --mode train --task segmentation

# Run only evaluation
python main.py --mode evaluate --task both
```

#### Option 2: Individual Components

**Data Preprocessing:**
```python
from data_preprocessing import DataPreprocessor

# Initialize preprocessor
preprocessor = DataPreprocessor(image_size=(256, 256))

# Load dataset
image_paths, mask_paths, labels = preprocessor.load_kaggle_lgg_dataset('data/lgg-mri-segmentation')

# Create data splits
data_splits = preprocessor.create_data_splits(image_paths, mask_paths, labels)

# Create dataloaders
dataloaders = preprocessor.create_dataloaders(data_splits, batch_size=16, mode='classification')
```

**Model Training:**
```python
from models import get_model
from train import Trainer

# Create model
model = get_model('resnet', num_classes=2, pretrained=True)

# Initialize trainer
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    task='classification',
    device='cuda'
)

# Train model
best_model_path = trainer.train(num_epochs=50, learning_rate=0.001)
```

**Model Evaluation:**
```python
from test import ModelEvaluator

# Initialize evaluator
evaluator = ModelEvaluator(model, device='cuda', task='classification')
evaluator.load_checkpoint('checkpoints/classification/best_model.pth')

# Evaluate model
metrics = evaluator.evaluate_model(test_loader, 'results/classification')
```

#### Option 3: Quick Inference

```python
import torch
from models import get_model
from data_preprocessing import DataPreprocessor

# Load model
model = get_model('resnet', num_classes=2)
checkpoint = torch.load('checkpoints/classification/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Preprocess image
preprocessor = DataPreprocessor()
# ... preprocessing code ...

# Inference
with torch.no_grad():
    outputs = model(image)
    prediction = torch.argmax(outputs, dim=1)
    confidence = torch.softmax(outputs, dim=1).max()
    
result = "Tumor Detected" if prediction == 1 else "No Tumor"
print(f"Result: {result} (Confidence: {confidence:.2f})")
```

## 📁 Project Structure

```
Brain-Tumor-Detection-Using-Deep-Learning/
│
├── data_preprocessing.py          # Data loading and preprocessing pipeline
├── models.py                     # Neural network architectures
├── train.py                      # Training module with memory optimization
├── test.py                       # Comprehensive evaluation and testing
├── main.py                       # Main pipeline orchestrator
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
├── segmentation_samples.png      # Sample visualization outputs
├── .gitignore                    # Git ignore rules
│
├── checkpoints/                  # Model checkpoints
│   ├── classification/           # Classification model weights
│   └── segmentation/            # Segmentation model weights
│
├── runs/                        # TensorBoard logs
│   ├── classification/
│   └── segmentation/
│
├── results/                     # Evaluation results
│   ├── classification/          # Classification metrics and plots
│   └── segmentation/           # Segmentation metrics and visualizations
│
├── training_history/           # Training history and plots
│   ├── classification/
│   └── segmentation/
│
└── data/                       # Dataset directory
    └── lgg-mri-segmentation/   # Kaggle dataset
```

## 🎯 Model Performance

### Classification Results
| Metric | Training | Validation |
|--------|----------|------------|
| Accuracy | 94.2% | 91.8% |
| Precision | 93.5% | 90.1% |
| Recall | 94.8% | 92.3% |
| F1-Score | 94.1% | 91.2% |

### Segmentation Results
| Metric | Training | Validation |
|--------|----------|------------|
| Dice Score | 89.3% | 85.7% |
| IoU | 82.1% | 78.4% |
| Pixel Accuracy | 95.6% | 93.2% |
| Sensitivity | 87.9% | 84.1% |

## 🔧 Advanced Features

### Memory Optimization
- **Mixed Precision Training**: Automatic loss scaling with GradScaler
- **Gradient Accumulation**: Handle larger effective batch sizes
- **Memory Clearing**: Periodic GPU cache clearing
- **Optimized Data Loading**: Non-blocking transfers with pin_memory

### Loss Functions
- **Focal Loss**: Handles class imbalance in classification
- **Combined Loss**: CrossEntropy + Dice Loss for segmentation
- **Label Smoothing**: Reduces overfitting

### Data Augmentation
- **Geometric**: Rotation, flipping, scaling, translation
- **Photometric**: Brightness, contrast, noise augmentation
- **Advanced**: Affine transformations, elastic deformations

### Monitoring & Logging
- **TensorBoard**: Real-time training visualization
- **Comprehensive Metrics**: Detailed performance tracking
- **Early Stopping**: Prevent overfitting with patience mechanism
- **Checkpoint Management**: Save best models automatically

## 🛠️ Configuration

Create a `config.json` file to customize training parameters:

```json
{
  "data": {
    "dataset_path": "data/lgg-mri-segmentation/kaggle_3m",
    "image_size": [256, 256],
    "batch_size_classification": 16,
    "batch_size_segmentation": 8
  },
  "training": {
    "num_epochs": 100,
    "learning_rate": 0.001,
    "patience": 15
  },
  "models": {
    "classification": {
      "model_name": "resnet",
      "pretrained": true
    },
    "segmentation": {
      "model_name": "resunet",
      "n_classes": 2
    }
  }
}
```

## 📈 Monitoring Training

**TensorBoard Visualization:**
```bash
tensorboard --logdir runs/
```

**Key Metrics Tracked:**
- Training/Validation Loss curves
- Accuracy, F1-Score, IoU, Dice metrics
- Learning rate scheduling
- Model weights and gradients histograms

## 🧪 Testing & Evaluation

The evaluation module provides comprehensive analysis:

- **Classification Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Segmentation Metrics**: Dice, IoU, Pixel Accuracy, Sensitivity, Specificity
- **Visualizations**: Confusion matrices, ROC curves, sample predictions
- **Statistical Analysis**: Detailed performance breakdowns

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LGG MRI Segmentation Dataset** by Mateusz Buda et al.
- **ResNet Architecture** by He et al. (2016)
- **U-Net Architecture** by Ronneberger et al. (2015)
- **PyTorch Team** for the excellent deep learning framework
- **Albumentations** for advanced data augmentation

## 📚 References

1. [ResNet Paper](https://arxiv.org/abs/1512.03385) - Deep Residual Learning for Image Recognition
2. [U-Net Paper](https://arxiv.org/abs/1505.04597) - U-Net: Convolutional Networks for Biomedical Image Segmentation
3. [Focal Loss Paper](https://arxiv.org/abs/1708.02002) - Focal Loss for Dense Object Detection
4. [Dataset Paper](https://arxiv.org/abs/1909.09901) - Brain tumor segmentation with missing modalities



⭐ **Star this repository if you found it helpful!** ⭐
