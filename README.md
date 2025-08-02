# 🎭 Multimodal Emotion Recognition using Transformers

This project explores **multimodal emotion recognition** using facial and vocal cues from the [RAVDESS](https://zenodo.org/record/1188976) dataset. We evaluate and compare three fusion strategies built on transformer and attention-based architectures to enhance affective computing systems that better understand human emotions.

## 🚀 Overview

Emotion recognition is central to improving human-computer interaction, sentiment analysis, and adaptive interfaces. Our system processes both **facial expressions (video)** and **speech signals (audio)** using:

- A **vision branch** powered by EfficientFace (pre-trained on AffectNet)
- An **audio branch** based on MFCCs and convolutional layers
- Three distinct **modality fusion strategies**:
  - Late Transformer Fusion
  - Intermediate Transformer Fusion
  - Intermediate Attention-Based Fusion (best performance)

Our best model achieved:
- **Top-5 Accuracy:** 98.12%

---

## 🗂️ Project Structure

├── data/ # Preprocessed RAVDESS dataset  
├── models/ # PyTorch models and architecture definitions  
├── utils/ # Preprocessing, dataloaders, and utilities  
├── experiments/ # Training scripts and result logs  
├── notebooks/ # Exploratory analysis and visualizations  
├── results/ # Output metrics and model predictions  
└── README.md # Project documentation

---

## 📊 Dataset

We used the **Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)**:

- 24 actors (12 female, 12 male)
- 7,356 audio-video recordings
- 8 emotions: neutral, calm, happy, sad, angry, fearful, surprise, and disgust
- Available in three formats: audio-only, video-only, and audiovisual

### Preprocessing:

- 🎧 **Audio:** Resampled to 16kHz, mono conversion, MFCC extraction, amplitude normalization
- 📹 **Video:** Extracted 15 frames per video, resized to 224×224, augmented using OpenFace
- 🔄 Synced and standardized for transformer compatibility

---

## 🧠 Model Architecture

### 🔍 Feature Extraction

- **Vision Branch**: 
  - EfficientFace (pretrained) + Temporal 1D Conv Layers
- **Audio Branch**: 
  - MFCC input → 4 Conv blocks → Global Avg Pooling

### 🔗 Fusion Mechanisms

| Fusion Type                  | Description                                                                 |
|-----------------------------|-----------------------------------------------------------------------------|
| Late Transformer Fusion      | Independent processing; fused at transformer stage                         |
| Intermediate Transformer     | Fusion at mid-feature layers with transformer-based cross-attention        |
| Intermediate Attention 🏆    | Scaled dot-product attention between modalities (no feature entanglement)  |

---

## 🧪 Experiments

### 🛠️ Training Setup

- Optimizer: SGD (lr = 0.04, momentum = 0.9, weight decay = 1e-3)
- Epochs: 100
- Data Augmentation: Random horizontal flips, rotations

### 📈 Performance Summary

| Method                      | Loss   | Top-1 Precision (%) | Top-5 Precision (%) |
|----------------------------|--------|---------------------|---------------------|
| Late Transformer Fusion    | 16.699 | 14.375              | 59.167              |
| Intermediate Transformer   | 35.392 | 13.958              | 85.208              |
| Intermediate Attention 🏆  | **2.393** | **33.958**         | **98.125**          |

---

## 📌 Key Insights

- Transformer-based fusion offers solid cross-modal alignment but risks overfitting.
- Simpler attention-based fusion (without direct fusion) performs best in this setup.
- Audio and facial features complement each other; their joint learning improves robustness.

---

## 📚 References

- Vaswani et al., *Attention is All You Need*
- Kumar et al., *Multimodal Emotion Recognition on RAVDESS*
- Yue et al., *Multi-task learning for emotion and intensity recognition*
- Sherman et al., *Speech Emotion Recognition with BLSTM + Attention*

---

## 🏁 Future Work

- Integrate **self-supervised embeddings** (e.g., Wav2Vec 2.0)
- Expand to larger datasets (IEMOCAP, CREMA-D)
- Explore **graph-based fusion** or **temporal transformers**
- Real-time deployment in adaptive user interfaces

---

## 📬 Contact

**Venkata Revanth Jyothula**  
📍 New York City  
📫 jyorevanth@gmail.com  
