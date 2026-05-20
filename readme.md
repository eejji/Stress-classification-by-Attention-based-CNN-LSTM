# Mental Stress Classification via Attention-Based CNN-LSTM on ECG Signals
<div align="center">

[![IEEE](https://img.shields.io/badge/IEEE-Published-blue?logo=ieee)](https://ieeexplore.ieee.org/abstract/document/10903250)
[![Conference](https://img.shields.io/badge/ICMLA-2024-orange)](https://www.icmla-conference.org/)

**Published at the 2024 IEEE International Conference on Machine Learning and Applications (ICMLA)**  
December 18–20, 2024 · Miami, FL, USA  
DOI: [10.1109/ICMLA61862.2024.00211](https://ieeexplore.ieee.org/abstract/document/10903250)

</div>

## Overview

This project presents a deep learning approach to **mental stress classification from ECG signals**, combining single-beat and multi-beat (rhythm) analysis through a novel **Attention-based CNN-LSTM** architecture.

Most prior work relied on either a single ECG cycle or a continuous cycle — but not both. This model fuses both representations in parallel to capture both short-term waveform features and long-term temporal patterns, achieving state-of-the-art performance.

## Key Contribution

> **Problem:** Using only a single ECG cycle misses long-term temporal patterns; using only a continuous cycle misses fine-grained waveform morphology.  
> **Solution:** A parallel dual-branch architecture that simultaneously learns from both Beat ECG and Rhythm ECG, fused through independent CNN → LSTM → Attention pipelines per branch.

**Keywords:** Stress · Electrocardiogram · CNN · LSTM · Attention Mechanism · Deep Learning

| Item | Detail |
|------|--------|
| **Task** | 3-class mental stress classification (Excitement / Neutral / Stress) |
| **Input** | ECG signal (single-beat + rhythm) |
| **Dataset** | [DREAMER](https://ieeexplore.ieee.org/document/7887697) |
| **Evaluation** | 5-fold cross-validation |
| **Accuracy** | **97%** |
| **F1-Score** | **0.969** |

---

#### **Abstract:**  
Stress is a state of tension felt when exposed to a difficult situation, and excessive stress can lead to chronic diseases, method to diagnose it early is needed. Electrocardiogram (ECG) signals reflect human physiological phenomena and can be easily obtained in a noninvasive manner, which can efficiently diagnose stress. Recent studies using ECG to diagnose stress tend to use only a single or continuous cycle of ECG signals. However, if only a single cycle is used, there is a problem that the characteristics of the continuous cycle cannot be analyzed, and vice versa, the same problem arises. To solve this problem, this study proposes an Attention-based CNN-LSTM model that uses a single cycle and continuous cycle of ECG together to diagnose stress. Using a single cycle and a continuous cycle together improves the stress classification performance because it learns the long and short-term patterns of the ECG. In addition, the model in this study uses a parallel structure Convolutional neural network (CNN) to extract and combine local features of single cycles and continuous cycles, and then highlights temporal patterns and important details through Long short-term memory (LSTM) and attention mechanisms to accurately identify physiological changes in complex ECG signals. Experiments on three multi-classes using the DREAMER database have achieved an average accuracy of 97% and an average f1 score of 0.969 and shown outstanding stress analysis efficiency of the proposed model. This approach shows higher performance and more accurate stress diagnosis when using both cycles together than when using a single cycle or a continuous cycle alone.

##### **Keywords:**  Stress, Electrocardiogram, Convolutional Neural Network, Long short-term memory, Attention mechanism, Deep learning 


## Pipeline

![Workflow](https://github.com/eejji/Stress-classification-by-Attention-based-CNN-LSTM/blob/main/image/Flowchart.png)

---

## Dataset: DREAMER

The [DREAMER dataset](https://ieeexplore.ieee.org/document/7887697) contains ECG recordings from **23 subjects** who watched emotion-eliciting film clips. Each subject rated their emotional state on Valence and Arousal scales (1–5).

Class labels are assigned using the **Valence-Arousal 2D emotion model**:

| Class | Valence | Arousal | Quadrant |
|-------|---------|---------|----------|
| **Excitement** | > 3 | > 3 | High Arousal, High Valence (Q1) |
| **Stress** | < 3 | > 3 | High Arousal, Low Valence (Q2) |
| **Neutral** | — | — | ECG baseline (resting state) |

The raw DREAMER `.mat` file is parsed by `Create_dataset.ipynb`, which extracts, denoises, normalizes, and saves a preprocessed `.pkl` file per subject.

---

## Preprocessing

Three-stage Butterworth filtering (order=2) to clean the raw ECG signal:

1. **High-pass filter** — 0.5 Hz cutoff (baseline wander removal)
2. **Band-stop filter** — 57–63 Hz (powerline noise removal)
3. **Low-pass filter** — 100 Hz cutoff (high-frequency noise removal)

After filtering, each signal is normalized with **MinMax scaling** to [0, 1].  
R-peaks are detected using **NeuroKit2** (`method='neurokit'`, sampling rate = 256 Hz).

![Preprocessing](https://github.com/eejji/Stress-classification-by-Attention-based-CNN-LSTM/blob/main/image/Preprocessing.png)

---

## Segmentation

Two complementary signal windows are extracted per R-peak:

| Type | Samples | Duration | Description |
|------|---------|----------|-------------|
| **Beat ECG** | 163 | ~0.64 s | Single cardiac cycle (61 samples pre-R, 102 samples post-R) |
| **Rhythm ECG** | 1280 | 5 s | 102 samples post-R based on R-peak, 5 seconds |

Beat and Rhythm segments are aligned by index so each sample pair feeds into the dual-branch model simultaneously.

![Segmentation](https://github.com/eejji/Stress-classification-by-Attention-based-CNN-LSTM/blob/main/image/Segmentation.png)

---

## Model Architecture

![Model](https://github.com/eejji/Stress-classification-by-Attention-based-CNN-LSTM/blob/main/image/Proposed_CNN_LSTM.jpg)

```
Beat ECG  ──► CNN Blocks (×3)  ─┐
                                  ├──► Fusion ──► LSTM ──► Attention ──► FC ──► Class
Rhythm ECG ──► CNN Blocks (×9) ─┘
```

**Convolution Block (shared building block)**
- Two 1D Convolutions + BatchNorm + ReLU
- Residual connection + MaxPooling for training stability

**Branch design**
- Beat branch: 3 convolution blocks (local waveform features)
- Rhythm branch: 9 convolution blocks (long-range temporal features)

**Post-fusion**
- LSTM captures sequential dependencies across the fused feature space
- Attention layer highlights the most diagnostically relevant time steps
- Fully-connected layer outputs 3-class probabilities

---

## Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adam |
| Learning rate | 0.0001 |
| Loss | Categorical cross-entropy |
| Batch size | 32 |
| Epochs | 60 |
| Early stopping | patience=15, min_delta=0.005, monitor=val_loss |
| Cross-validation | 10-fold Stratified K-Fold (random_state=42) |
| Random seed | 12 (NumPy) |

---

## Results

10-fold stratified cross-validation on the DREAMER database:

![Performance Table](https://github.com/eejji/Stress-classification-by-Attention-based-CNN-LSTM/blob/main/image/Performance_table.png)

| Model | Accuracy | F1-Score |
|-------|----------|----------|
| Beat only | 70.3% | 0.7 |
| Rhythm only | 91.5% | 0.914 |
| **Fusion + Attention (proposed)** | **97%** | **0.969** |

- **+26.7% accuracy** and **+0.265 F1** over single-beat baseline
- **+5.5% accuracy** and **+0.055 F1** over rhythm-only baseline
- All classes exceeded **97% precision** in the best fold

Evaluation metrics reported per fold: Accuracy, Precision, Recall, Specificity, F1-Score, Confusion Matrix.

---

## Project Structure

```
.
├── Create_dataset.ipynb   # Parse DREAMER .mat → denoised .pkl
├── FusionNetwork.ipynb    # Training loop with 10-fold CV
├── Model_selection.py     # Model definition (fusion_model, DotProductAttention)
├── data_utils.py          # Signal filtering, R-peak detection, segmentation, normalization
├── Plot_utils.py          # Learning curves and confusion matrix visualization
├── requirements.txt       # Python dependencies
└── image/                 # Figures used in README and paper
```

---

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Prepare Dataset

Download the [DREAMER dataset](https://ieeexplore.ieee.org/document/7887697) (`.mat` format) and update the path in `Create_dataset.ipynb`:

```python
mat_path = "path/to/DREAMER.mat"
save_path = "path/to/Denoised.pkl"
```

Run all cells in `Create_dataset.ipynb` to generate the preprocessed pickle file.

### 2. Train the Model

Update the paths in `FusionNetwork.ipynb`:

```python
pkl_path    = "path/to/Denoised.pkl"
model_save_path = "path/to/save/models/"
```

Run all cells to execute 10-fold cross-validation. Each fold saves the best model as `.h5` and prints accuracy, F1-score, and the confusion matrix.

---

## Citation

```bibtex
@inproceedings{lee2024mental,
  title={Mental Stress Classification by Attention-Based CNN-LSTM Algorithm of Electrocardiogram Signal},
  author={Lee, Jihun and Hong, Jisun and Choi, Daegil and Jung, Jaehyo},
  booktitle={2024 International Conference on Machine Learning and Applications (ICMLA)},
  pages={1356--1361},
  year={2024},
  organization={IEEE}
}
```
