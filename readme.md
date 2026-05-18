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


## Work flow
![image](https://github.com/eejji/Stress-classification-by-Attention-based-CNN-LSTM/blob/main/image/Flowchart.png)


## Preprocessing 
Three-stage filtering to clean the raw ECG signal:

1. **Low-pass filter** — 100 Hz cutoff
2. **High-pass filter** — 0.5 Hz cutoff
3. **Notch filter** — 57–63 Hz (powerline noise removal)

![Preprocessing](https://github.com/eejji/Stress-classification-by-Attention-based-CNN-LSTM/blob/main/image/Preprocessing.png)

## Segmentation
Two complementary signal windows are extracted per subject:

| Type | Duration | Description |
|------|----------|-------------|
| **Beat ECG** | 0.64 s | Single cardiac cycle (0.24 s pre-R, 0.40 s post-R) |
| **Rhythm ECG** | 10 s | Continuous multi-cycle window |
  
![Segmentation](https://github.com/eejji/Stress-classification-by-Attention-based-CNN-LSTM/blob/main/image/Segmentation.png)


## Proposed Attention-based CNN-LSTM
![image](https://github.com/eejji/Stress-classification-by-Attention-based-CNN-LSTM/blob/main/image/Proposed_CNN_LSTM.jpg)
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

## Result
This project evaluates model performance using 5-fold cross-validation.   
The performance of individual networks (Beat, Rhythm) is compared with that of the combined Fusion models.   

| Model | Accuracy | F1-Score |
|-------|----------|----------|
| Beat only | 70.3% | 0.7 |
| Rhythm only | 91.5% | 0.914 |
| **Fusion + Attention (proposed)** | **97%** | **0.969** |

- **+26.7% accuracy** and **+0.265 F1** over single-beat baseline
- **+5.5% accuracy** and **+0.055 F1** over rhythm-only baseline
- All classes exceeded **97% precision** in the best fold

---
![image](https://github.com/eejji/Stress-classification-by-Attention-based-CNN-LSTM/blob/main/image/Performance_table.png)


## Conclusion
This study proposes an Attention-based CNN-LSTM model that uses a single cycle and a continuous cycle of ECG together for stress diagnosis. The proposed model integrates single and continuous ECG cycles through its Beat and Rhythm networks, which operate in parallel. By utilizing repeated convolution blocks for feature extraction and enhancing pattern recognition with LSTM and Attention mechanisms, the model demonstrates comprehensive learning analysis. We achieved an average accuracy of 97% and an average F1 score of 0.969 as a result of the 5-fold cross validation performance evaluation. The accuracy increased by about 22% and the F1 score increased by 0.21 compared to the case of using only a single cycle, and the accuracy increased by about 7% and the F1 score increased by 0.07 compared to the case of using only a continuous cycle. In the 5th Fold of the cross-validation, all classes showed high predictive performance of more than 97%. These results demonstrate improved performance when single and continuous cycles are used together, as opposed to using either single-cycle or continuous-cycle methods. In future studies, we plan to improve accuracy and reliability by devising model weight reduction, model interpretability, and generalization verification methods.


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
