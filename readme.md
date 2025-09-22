# Mental Stress Classification by Attention-Based CNN-LSTM Algorithm of Electrocardiogram Signal

## Conference / Publication
**Conference:** 2024 International Conference on Machine Learning and Applications (ICMLA)   
**Date:** 18-20 December 2024   
**Location:** Miami, FL, USA   
**DOI:** [10.1109/ICMLA61862.2024.00211](https://ieeexplore.ieee.org/abstract/document/10903250)

## Task- At A Glance:
Through the Attention-based CNN-LSTM model, single and continuous-cycle ECG feature are extracted and fused to diagnose mental stress.  
1. __Task__: Diagnosis Stress
2. __Input__: ECG Signal (single and continuous-cycle)
3. __Output__:  3 Class (Excitement, Neutral, Stress)
4. __Database__: (1) [DREAMER](https://ieeexplore.ieee.org/document/7887697)
5. __Preprocessing__: Butterworth filter denoising, Normalization, Segmentation
6. __Fusion method__: Concatenate
7. __Result__: Accuracy: **97 %**, F1-score **0.969** (3 class)

#### **Abstract:**  
Stress is a state of tension felt when exposed to a difficult situation, and excessive stress can lead to chronic diseases, method to diagnose it early is needed. Electrocardiogram (ECG) signals reflect human physiological phenomena and can be easily obtained in a noninvasive manner, which can efficiently diagnose stress. Recent studies using ECG to diagnose stress tend to use only a single or continuous cycle of ECG signals. However, if only a single cycle is used, there is a problem that the characteristics of the continuous cycle cannot be analyzed, and vice versa, the same problem arises. To solve this problem, this study proposes an Attention-based CNN-LSTM model that uses a single cycle and continuous cycle of ECG together to diagnose stress. Using a single cycle and a continuous cycle together improves the stress classification performance because it learns the long and short-term patterns of the ECG. In addition, the model in this study uses a parallel structure Convolutional neural network (CNN) to extract and combine local features of single cycles and continuous cycles, and then highlights temporal patterns and important details through Long short-term memory (LSTM) and attention mechanisms to accurately identify physiological changes in complex ECG signals. Experiments on three multi-classes using the DREAMER database have achieved an average accuracy of 97% and an average f1 score of 0.969 and shown outstanding stress analysis efficiency of the proposed model. This approach shows higher performance and more accurate stress diagnosis when using both cycles together than when using a single cycle or a continuous cycle alone.

##### **Keywords:**  Stress, Electrocardiogram, Convolutional Neural Network, Long short-term memory, Attention mechanism, Deep learning 

##### Work flow
![image](<img width="14923" height="3536" alt="Image" src="https://github.com/user-attachments/assets/779e19ce-f524-45d4-9254-027504ca0ab2" />)
