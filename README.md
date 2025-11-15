# 📰 Sentiment Classification of News Articles Using BERT
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![HuggingFace](https://img.shields.io/badge/Model-BERT%20(Hierarchical)-yellow)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-red)

This repository contains the complete implementation of the research paper titled:

**"Sentiment Classification of News Articles Using BERT"**

📍 **Presented at:** *International Conference on Digital Technologies for Business Excellence and Sustainable Development and Creating Viksit Bharat@2047 (ICDTBESDVB 2025)*  
📅 **Venue:** IIT (ISM) Dhanbad  
🗓️ **Date:** 5th–6th July 2025

---

## 📌 Project Overview

This project classifies news articles into three sentiment categories: **Positive**, **Negative**, and **Neutral**.

A standard 3-class BERT model often shows high confusion between the "Positive" and "Neutral" categories. This project implements an advanced **hierarchical two-model pipeline** to solve this problem, resulting in a more accurate and nuanced classifier that understands subtle context in news headlines.

The dataset was enhanced using advanced data augmentation techniques, including:

- **Synonym Replacement**
- **Back-Translation**

These techniques expand dataset diversity and improve generalization.

---

## 🧠 Model Architecture

This project uses a two-stage hierarchical pipeline. Instead of one model handling all three classes, two specialist models work together.

### 🔻 Model 1: The "Sentiment Detector"
- **Purpose:** Determine whether the text is `Negative` or `Not-Negative`
- **Base Model:** `bert-base-uncased`
- **Hugging Face Repo:**  
  👉 https://huggingface.co/SourabhKhamankar/model-1-neg-vs-notneg

### 🔺 Model 2: The "Nuance Detector"
- **Purpose:** If Model 1 predicts `Not-Negative`, this model distinguishes between `Neutral` and `Positive`
- **Base Model:** `bert-base-uncased`
- **Hugging Face Repo:**  
  👉 https://huggingface.co/SourabhKhamankar/model-2-neu-vs-pos

This modular design greatly improves recognition of subtle positive sentiment.

---

## 📊 Results and Performance

### 1. The Initial Problem (Single 3-Class Model)

A single BERT model achieved **92% accuracy**, but performed poorly on the **Positive** class.

| **Class**       | **Precision** | **Recall** | **F1-Score** | **Support** |
|-----------------|---------------|------------|--------------|-------------|
| Negative (0)    | 0.86          | 0.96       | 0.91         | 56          |
| Neutral (1)     | 0.94          | 0.96       | 0.95         | 248         |
| Positive (2)    | 0.84          | 0.63       | 0.72 ⚠️      | 43          |

### **Overall Metrics**

| **Metric**        | **Score** |
|-------------------|-----------|
| Accuracy          | 92%       |
| Macro F1-Score    | 0.86      |
| Weighted F1-Score | 0.92      |

> ❗ The **Positive class F1-score of 0.72** indicated high confusion with Neutral.
## Baseline 3-Class Model — Confusion Matrix

![3-Class Confusion Matrix](images/Single%203-Class%20Model%20Confusion%20Matrix.png)

---

## 🚀 Hierarchical Model Performance

By splitting the task into two specialized models, positive sentiment detection improved significantly.

### 🧭 Model 1 — “Sentiment Detector” (Negative vs Not-Negative)

| **Class**          | **Precision** | **Recall** | **F1-Score** | **Support** |
|--------------------|---------------|------------|--------------|-------------|
| Negative (0)       | 0.87          | 0.95       | 0.91         | 56          |
| Not-Negative (1)   | 0.99          | 0.97       | 0.98         | 291         |

### **Overall Metrics**

| **Metric**        | **Score** |
|-------------------|-----------|
| Accuracy          | 0.97      |
| Macro F1-Score    | 0.94      |
| Weighted F1-Score | 0.97      |
## Model 1 — Negative vs Not-Negative Confusion Matrix

![Model 1 Confusion Matrix](images/Sentiment%20Detector%20Confusion%20Matrix.png)


---

## 🧠 Model 2 — “Nuance Detector” (Neutral vs Positive)

| **Class**       | **Precision** | **Recall** | **F1-Score** | **Support** |
|-----------------|---------------|------------|--------------|-------------|
| Neutral (0)     | 0.95          | 0.99       | 0.97         | 248         |
| Positive (1)    | 0.94          | 0.72       | 0.82 ⭐      | 43          |

### **Overall Metrics**

| **Metric**        | **Score** |
|-------------------|-----------|
| Accuracy          | 0.95      |
| Macro F1-Score    | 0.89      |
| Weighted F1-Score | 0.95      |
## Model 2 — Neutral vs Positive Confusion Matrix

![Model 2 Confusion Matrix](images/Nuance%20Detector%20Confusion%20Matrix.png)


---

### ✅ Final Outcome

**Positive class F1-score improved from 0.72 → 0.82**, effectively solving the key weakness of the base model.  
This confirms the **hierarchical BERT pipeline** is more accurate, reliable, and context-aware.

---

## 🌐 Live Demo

Try the deployed model here:  
👉 **https://bert-news-sentiment-classifier-egsu3ojnmntfotixdxvq35.streamlit.app/**

---

## 🛠️ Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/SourabhKhamankar22/BERT-news-sentiment-classifier.git
cd BERT-news-sentiment-classifier
```
### 2️⃣ Create and activate a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```
### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```
*Your `requirements.txt` should contain:*
```bash 
streamlit
torch
transformers
sentencepiece
sacremoses
pandas
```

## 🚦 Usage

### A. Running the Live App Locally
```bash
streamlit run app.py
```
* The app will open in your browser at http://localhost:8501
* It automatically downloads the two models from Hugging Face Hub.

### 📓 B. Project Notebooks (Full Development Pipeline)
This repository includes the full research and development process in Jupyter Notebooks:
1.  `newssentimentanalysis.ipynb`: The initial exploration, data augmentation (3-hour process), and training of the first flawed 3-class model.
2.  `model_1.ipynb`: The notebook used to train and evaluate the "Sentiment Detector" (Model 1).
3.  `model_2.ipynb`: The notebook used to train and evaluate the "Nuance Detector" (Model 2).
4.  `final_predictor.ipynb`: A testing notebook to load the two final models and run test predictions.

### 🧩 C. Using the Model Pipeline in Python
Here is how to use the final two-model pipeline in any Python script:

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load Model 1 (The "Sentiment Detector") ---
model_1_name = "SourabhKhamankar/model-1-neg-vs-notneg"
tokenizer_1 = AutoTokenizer.from_pretrained(model_1_name)
model_1 = AutoModelForSequenceClassification.from_pretrained(model_1_name).to(device)
model_1.eval()

# --- Load Model 2 (The "Nuance Detector") ---
model_2_name = "SourabhKhamankar/model-2-neu-vs-pos"
tokenizer_2 = AutoTokenizer.from_pretrained(model_2_name)
model_2 = AutoModelForSequenceClassification.from_pretrained(model_2_name).to(device)
model_2.eval()


# --- Prediction Function ---
def predict_sentiment(text: str):
    """
    Runs the hierarchical prediction:
    1. Check if Negative vs. Not-Negative (using Model 1).
    2. If Not-Negative, check if Neutral vs. Positive (using Model 2).
    """
    
    # --- Step 1: Feed text to Model 1 ---
    inputs_1 = tokenizer_1(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs_1 = {k: v.to(device) for k, v in inputs_1.items()}
    
    with torch.no_grad():
        outputs_1 = model_1(**inputs_1)
    
    pred_1 = torch.argmax(outputs_1.logits, dim=1).item()
    
    # --- Step 2: Check Model 1's prediction (0=Negative, 1=Not-Negative) ---
    if pred_1 == 0:
        probs_1 = torch.nn.functional.softmax(outputs_1.logits, dim=-1)
        confidence = probs_1[0][pred_1].item()
        return {"sentiment": "Negative", "confidence": round(confidence, 4)}
    
    else:
        # Model 1 predicted "Not-Negative" (1), so we run Model 2
        
        # --- Step 3: Feed text to Model 2 ---
        inputs_2 = tokenizer_2(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs_2 = {k: v.to(device) for k, v in inputs_2.items()}
    
        with torch.no_grad():
            outputs_2 = model_2(**inputs_2)
            
        pred_2 = torch.argmax(outputs_2.logits, dim=1).item()
        probs_2 = torch.nn.functional.softmax(outputs_2.logits, dim=-1)
        confidence = probs_2[0][pred_2].item()
        
        # --- Step 4: Check Model 2's prediction (0=Neutral, 1=Positive) ---
        if pred_2 == 0:
            return {"sentiment": "Neutral", "confidence": round(confidence, 4)}
        else: # pred_2 == 1
            return {"sentiment": "Positive", "confidence": round(confidence, 4)}

# Example
text = "The company reported record profits and announced a special dividend for shareholders."
prediction = predict_sentiment(text)
print(prediction)
# Output: {'sentiment': 'Positive', 'confidence': 0.9342}
```

## 📉 Limitations
While the hierarchical pipeline improved performance significantly, the model still faces several limitations:

### 1. Limited Dataset Size & Imbalance
The Positive and Neutral classes contain fewer and less diverse training samples.  
This often leads to confusion between these two categories.

### 2. Dataset Diversity & Bias
A narrow writing style may introduce keyword bias (e.g., “profit”, “loss”), reducing the model’s deeper semantic understanding.

### 3. Subjectivity of Sentiment
Human labeling inconsistencies affect how borderline sentences are interpreted.

### 4. Domain Shift / Specificity
Performance may drop on domain-specific news (finance, legal, scientific).

### 5. Hierarchical Pipeline Complexity
Two models increase inference cost and error propagation.

### 6. Limited Context of Headlines
Short headlines lack context, making sentiment harder to interpret.

---

## 🔭 Future Scope
This project opens several possibilities for future improvements:

### 1. Larger, More Balanced Dataset
Collecting more Positive and Neutral samples would further reduce confusion.

### 2. Using Advanced Transformer Models
Models like **RoBERTa, DeBERTa, ELECTRA, BERT-Large** may capture subtle cues better.

### 3. Domain-Specific Fine-Tuning
Specialized nuance detectors for finance, sports, politics, etc.

### 4. Unified Pipeline via Knowledge Distillation
A single distilled model for lower latency and simpler deployment.

### 5. Multilingual Support
Using **mBERT/XLM-R** to classify news globally.

### 6. Multi-Label Sentiment Classification
Extract multiple sentiments from a single headline or article.

---

## 🏁 Conclusion
This project demonstrates that a traditional 3-class BERT model is insufficient for subtle news sentiment classification—especially for distinguishing **Positive** from **Neutral** headlines (baseline F1: **0.72**).

To solve this, a hierarchical two-stage pipeline was built:

- **Model 1:** Detects *Negative vs Not-Negative*  
- **Model 2:** Detects *Neutral vs Positive*

This architecture significantly improved class-wise performance:

- **Negative:** F1 = **0.91**, Recall = **0.95**  
- **Neutral:** F1 = **0.97**, Recall = **0.99**  
- **Positive:** F1 improved from **0.72 → 0.82**

With model accuracies of **0.97 (Model 1)** and **0.95 (Model 2)**, the final classifier is robust, context-aware, and reliable for real-world news sentiment analysis.

With larger datasets and modern transformer architectures, this system can evolve into a highly scalable solution for news analytics and media intelligence.


## 📚 Citation
If you use this work, please cite it as:
```bash
@inproceedings{Khamankar2025SentimentBERT,
  title     = {Sentiment Classification of News Articles Using BERT},
  author    = {Sourabh Khamankar and et al.},
  booktitle = {Proceedings of ICDTBESDVB 2025, IIT (ISM) Dhanbad},
  year      = {2025}
}
```


## ✅ Notes / Remaining Tips

- Ensure your local Python version is >=3.10 for compatibility.

- Use GPU if available for faster inference.

- Optional: Install hf_xet package for faster downloads from Hugging Face Hub:
```
pip install hf_xet
```  