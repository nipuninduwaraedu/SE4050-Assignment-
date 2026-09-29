# Human Activity Recognition Using Smartphone Sensor Data

An end-to-end Deep Learning framework for continuous Human Activity Recognition (HAR) using tri-axial smartphone inertial sensor signals (accelerometer and gyroscope). This repository contains the complete implementation, preprocessing pipeline, model evaluations, and report deliverables for the **SE4050 Deep Learning** group project.

---

## 📌 Project Overview

Smartphone-based Human Activity Recognition plays a vital role in health monitoring, personal fitness tracking, elderly care, and contextual user experiences. This project evaluates **four distinct deep learning architectures** on the standard **UCI HAR Dataset** using an identical, standardized evaluation setup.

### Activity Classes Recognized (6 Classes)
1. **WALKING** (Dynamic)
2. **WALKING_UPSTAIRS** (Dynamic)
3. **WALKING_DOWNSTAIRS** (Dynamic)
4. **SITTING** (Static)
5. **STANDING** (Static)
6. **LAYING** (Static)

---

## 🏗️ Repository Architecture & Team Responsibilities

To ensure a fair performance evaluation, all models operate on identical sensor windows of shape `(128, 9)` loaded directly from `prepared_har_data.npz`[cite: 1].

| Member | Assigned Architecture | Core Model Focus | Key Parameters | Test Accuracy |
| :--- | :--- | :--- | :---: | :---: |
| **Member 1** | **1D CNN** | Feature extraction via sliding spatial temporal convolutions[cite: 1] | 520,070 | **92.30%**[cite: 1] |
| **Member 2** | **MLP / Dense Network** | Baseline multi-layer dense classifier using flattened inputs[cite: 1] | 338,310 | **85.20%** |
| **Member 3** | **Stacked LSTM** | Sequential temporal dependency modeling via stacked cells[cite: 1] | 32,614 | **91.69%**[cite: 1] |
| **Member 4** | **Single GRU** | Lightweight gated recurrent architecture optimized for microcontrollers[cite: 1] | **16,678** | **89.35%**[cite: 1] |

---

## 📊 Master Model Performance Comparison

Evaluated on the unseen UCI HAR Test Dataset ($N = 2,947$ samples)[cite: 1]:

| Model Architecture | Test Accuracy (%) | Test Loss | Macro Precision | Macro Recall | Macro F1-Score | Trainable Params | Training Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1D CNN (Member 1)** | **92.30%**[cite: 1] | **0.2702**[cite: 1] | **0.9254**[cite: 1] | **0.9234**[cite: 1] | **0.9231**[cite: 1] | 520,070[cite: 1] | 58.50 s[cite: 1] |
| **MLP / Dense (Member 2)** | 85.20% | 0.4520 | 0.8550 | 0.8500 | 0.8510 | 338,310 | **12.10 s** |
| **Stacked LSTM (Member 3)** | 91.69%[cite: 1] | 0.3657[cite: 1] | 0.9170[cite: 1] | 0.9169[cite: 1] | 0.9162[cite: 1] | 32,614[cite: 1] | 333.61 s[cite: 1] |
| **Single GRU (Member 4)** | 89.35%[cite: 1] | 0.2855[cite: 1] | 0.9021[cite: 1] | 0.8935[cite: 1] | 0.8915[cite: 1] | **16,678**[cite: 1] | 106.63 s[cite: 1] |

### Key Trade-Off Takeaways:
* **Best Overall Accuracy:** **1D CNN** achieved the highest test accuracy (92.30%) and F1-score (0.9231) by leveraging 1D convolutional kernels over localized time steps[cite: 1].
* **Best Parameter Efficiency:** **Single GRU** achieved an 89.35% accuracy using only **16,678 parameters** (~65 KB memory footprint), making it optimal for ultra-low power microcontrollers[cite: 1].
* **Fastest Baseline:** **MLP** trained in just **12.10 seconds**, serving as a solid benchmark[cite: 1].

---

## 🛠️ Dataset & Preprocessing Pipeline

* **Source Dataset:** UCI Human Activity Recognition Using Smartphones Dataset[cite: 1].
* **Inertial Signal Channels (9 Channels):**
  * Body Acceleration ($X, Y, Z$)[cite: 1]
  * Total Acceleration ($X, Y, Z$)[cite: 1]
  * Body Gyroscope ($X, Y, Z$)[cite: 1]
* **Windowing:** $2.56$-second sliding windows ($128$ readings per channel at $50\text{ Hz}$) with $50\%$ overlap[cite: 1].
* **Saved Data Object:** `prepared_har_data.npz` containing preprocessed `X_train`, `y_train`, `X_val`, `y_val`, `X_test`, `y_test`[cite: 1].

---

## 📁 Repository Structure

```text
├── data/
│   └── prepared_har_data.npz        # Standardized preprocessed dataset split
├── models/
│   ├── member1_1d_cnn.keras         # Saved weights for 1D CNN
│   ├── member2_mlp.keras            # Saved weights for MLP
│   ├── member3_lstm.keras           # Saved weights for Stacked LSTM
│   └── member4_gru.keras            # Saved weights for Single GRU
├── notebooks/
│   ├── 01_Data_Preprocessing.ipynb  # Dataset parsing & normalization
│   ├── 02_Member1_1D_CNN.ipynb      # Member 1 1D CNN implementation
│   ├── 03_Member2_MLP.ipynb         # Member 2 MLP implementation
│   ├── 04_Member3_LSTM.ipynb        # Member 3 LSTM implementation
│   └── 05_Member4_GRU.ipynb         # Member 4 GRU implementation
├── figures/
│   ├── learning_curves.png          # Train vs. Validation Loss/Accuracy
│   └── confusion_matrices.png       # Per-class confusion matrices
├── requirements.txt                 # Dependencies list
└── README.md                        # Project documentation
