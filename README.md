# Extended LSTM: Adaptive Feature Gating for Toxic Comment Classification

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/4dd4c82e-5b98-4443-b5ef-23a226a2aee8" />

To ensure transparency and reproducibility, the full implementation of the proposed Extended LSTM (xLSTM) framework is made publicly available. The repository contains modularized code for dataset preprocessing, model training, ablation experiments, and evaluation pipelines.  Each module is designed for extensibility, allowing researchers to adapt the architecture for alternative tasks such as toxicity span prediction, multi-label sentiment analysis, and conversational moderation. The implementation integrates key innovations introduced in this work, including: Cosine-Similarity Gating:} Implements feature-dimension alignment using a learnable reference vector $\mathbf{v}$ to enhance minority-class gradient propagation. Hybrid Embedding Fusion: Supports joint integration of contextual (BERT-CLS), semantic (GloVe, FastText), and character-level BiLSTM features. Adaptive Loss Functions: Provides dynamic class weighting and focal loss for severe label imbalance. Lightweight Evaluation Suite: Enables real-time inference benchmarking with sub-50 ms latency using CPU-level optimization. Schematic Overview of the xLSTM Implementation Framework. The modular codebase encapsulates data handling, model configuration, training pipelines, and ablation utilities, providing a reproducible foundation for advanced sequence modeling research.

<!-- (Optional image preview or architecture image) -->

## 📌 Overview

This project presents a **robust toxic comment detection system** designed to identify harmful or abusive language in user-generated content on online platforms. By leveraging **high-dimensional linguistic features** and **a wide spectrum of machine learning models**, our approach addresses two major challenges in this domain:

* **Class imbalance**: Toxic comments are significantly underrepresented in most real-world datasets.
* **Short-text classification**: Many toxic inputs lack contextual depth, making accurate classification more complex.

The system integrates **classical, ensemble, and deep learning models**, providing a comparative study that helps balance **precision and recall**, minimizing both **over-censorship** and **false negatives**.

---

## 📊 Dataset

* **Source**: Kaggle (Jigsaw Toxic Comment Classification Challenge)
* **Size**: 159,571 total comments
* **Labeled Subset**: 1,000 manually annotated comments used for in-depth evaluation
* **Imbalance Note**: Toxic comments represent a small minority, reflecting real-world distribution.

---

## 🧱 Methodology

### 🔍 Feature Engineering

Each comment is represented by a **high-dimensional feature vector**:

$$
\mathbf{f}_i = \left[ \mathbf{e}_i, \mathbf{l}_i, \mathbf{p}_i, \mathbf{d}_i \right]
$$

Where:

* $\mathbf{e}_i$: BERT-based word embeddings
* $\mathbf{l}_i$: Lexicon-based toxicity and sentiment scores
* $\mathbf{p}_i$: One-hot encoded part-of-speech (POS) tags
* $\mathbf{d}_i$: Dependency parse features extracted from syntactic structures

All features are normalized to ensure compatibility across heterogeneous representations.

### 🧠 Models Evaluated

| Category      | Models                                       |
| ------------- | -------------------------------------------- |
| Classical     | Naive Bayes, Logistic Regression, Linear SVM |
| Ensemble      | Random Forest, XGBoost                       |
| Deep Learning | LSTM, Extended LSTM (xLSTM), BERT variants   |

Each model was trained on the extracted features using weighted binary cross-entropy to emphasize toxic labels.

---

## ⚙️ Training & Evaluation

### ✅ Loss Function

Custom weighted binary cross-entropy loss:

$$
\mathcal{L} = -\frac{1}{N} \sum_{i=1}^N w_i \left[ y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i) \right]
$$

* $w_i$ adjusted to prioritize toxic class
* $\hat{y}_i$: predicted probability
* $y_i$: ground truth

### 📈 Metrics Used

* **Recall**: Prioritizing detection of harmful content
* **Precision**: Avoiding over-censorship of benign content
* **F1-Score**: Balance between recall and precision
* **Confusion Matrix**: For error analysis

---

## 📊 Key Results

| Model               | Recall | F1-Score | FN (missed toxic) | FP (false toxic) |
| ------------------- | ------ | -------- | ----------------- | ---------------- |
| Logistic Regression | 0.48   | 0.48     | 358               | 702              |
| Linear SVM (SGD)    | 0.37   | 0.39     | 322               | 695              |
| Random Forest       | 0.43   | 0.42     | 1127              | 702              |
| XGBoost             | 0.42   | 0.42     | **1337**          | **82**           |

* **Linear SVM** is a strong baseline with balanced performance.
* **XGBoost** has high precision but low recall, missing many toxic cases.
* Deep models like **xLSTM** and **BERT** (in extended work) showed improved generalization but require more resources.

---

## 📉 Learning Curve Insights

* **XGBoost** shows the best generalization performance.
* **Random Forest** overfits the training data.
* **SVM** underfits, but with stable learning behavior.
* **BERT-based transformers** (future integration) promise enhanced context understanding.

---

## 🔬 Conclusions

This project illustrates:

* The **importance of rich feature engineering** in short-text classification.
* The **trade-offs between different model categories**.
* A framework capable of adapting to **real-world content moderation needs**.

---

## 🚀 Future Work

* Integration of **hybrid models**: combining ensemble and transformer-based architectures.
* Implementation of **data augmentation** and **contrastive learning** to improve robustness.
* Exploration of **multilingual toxic content detection** using pre-trained language models like mBERT or XLM-R.

---

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/csislam/toxic-comment-detection.git
cd toxic-comment-detection

# Install dependencies
pip install -r requirements.txt

# Run experiments
python train_model.py --model svm
```

---

## 📁 Project Structure

```
├── data/                  # Datasets and preprocessing
├── models/                # Model definitions (SVM, RF, LSTM, etc.)
├── outputs/               # Results, confusion matrices, and figures
├── train_model.py         # Main training script
├── utils.py               # Helper functions
├── README.md              # This file
└── requirements.txt       # Python dependencies
```
