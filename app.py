import streamlit as st
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- Page Config ---
st.set_page_config(
    page_title="News Sentiment Classifier",
    page_icon="📰",
    layout="wide"
)

# --- Model Loading ---
@st.cache_resource
def load_models():
    """Load both hierarchical models and tokenizers from Hugging Face."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model 1: "Sentiment Detector" (Negative vs. Not-Negative)
    model_1_name = "SourabhKhamankar/model-1-neg-vs-notneg"
    tokenizer_1 = AutoTokenizer.from_pretrained(model_1_name)
    model_1 = AutoModelForSequenceClassification.from_pretrained(model_1_name).to(device)
    model_1.eval()

    # Model 2: "Nuance Detector" (Neutral vs. Positive)
    model_2_name = "SourabhKhamankar/model-2-neu-vs-pos"
    tokenizer_2 = AutoTokenizer.from_pretrained(model_2_name)
    model_2 = AutoModelForSequenceClassification.from_pretrained(model_2_name).to(device)
    model_2.eval()

    return device, tokenizer_1, model_1, tokenizer_2, model_2

# Load models and show a spinner
with st.spinner("Loading AI Models... This may take a moment."):
    device, tokenizer_1, model_1, tokenizer_2, model_2 = load_models()

# --- Prediction Function ---
def predict_sentiment(text: str):
    """
    Runs the hierarchical prediction and returns the final sentiment,
    confidence, and a dictionary of all three probabilities.
    """
    
    # --- Step 1: Feed text to Model 1 (Neg vs. Not-Neg) ---
    inputs_1 = tokenizer_1(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs_1 = {k: v.to(device) for k, v in inputs_1.items()}
    
    with torch.no_grad():
        outputs_1 = model_1(**inputs_1)
    
    # Use .squeeze() to make sure it works for single inputs
    probs_1 = torch.nn.functional.softmax(outputs_1.logits, dim=-1).squeeze()
    prob_negative = probs_1[0].item()
    prob_not_negative = probs_1[1].item()
    
    # --- Step 2: Check Model 1's prediction ---
    if prob_negative > prob_not_negative:
        # Model 1 predicted "Negative"
        # We need to estimate the other two (they will be small)
        # This is an approximation, but necessary for the bar chart
        final_probs = {
            "Negative": prob_negative,
            "Neutral": prob_not_negative * 0.5,  # Split the "not-negative" prob
            "Positive": prob_not_negative * 0.5
        }
        final_sentiment = "Negative"
        final_confidence = prob_negative

    else:
        # Model 1 predicted "Not-Negative", so we run Model 2
        
        # --- Step 3: Feed text to Model 2 (Neu vs. Pos) ---
        inputs_2 = tokenizer_2(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs_2 = {k: v.to(device) for k, v in inputs_2.items()}
    
        with torch.no_grad():
            outputs_2 = model_2(**inputs_2)
            
        # Use .squeeze() to make sure it works for single inputs
        probs_2 = torch.nn.functional.softmax(outputs_2.logits, dim=-1).squeeze()
        prob_neutral_given_not_neg = probs_2[0].item()
        prob_positive_given_not_neg = probs_2[1].item()
        
        # --- Step 4: Calculate final probabilities ---
        # P(Neutral) = P(Not-Negative) * P(Neutral | Not-Negative)
        final_prob_neutral = prob_not_negative * prob_neutral_given_not_neg
        # P(Positive) = P(Not-Negative) * P(Positive | Not-Negative)
        final_prob_positive = prob_not_negative * prob_positive_given_not_neg
        
        final_probs = {
            "Negative": prob_negative,
            "Neutral": final_prob_neutral,
            "Positive": final_prob_positive
        }
        
        # Determine final winner
        if final_prob_neutral > final_prob_positive:
            final_sentiment = "Neutral"
            final_confidence = final_prob_neutral
        else:
            final_sentiment = "Positive"
            final_confidence = final_prob_positive

    return {
        "sentiment": final_sentiment, 
        "confidence": round(final_confidence, 4),
        "all_probs": final_probs
    }

# --- Streamlit UI ---
st.title("📰 News Sentiment Classifier")

# Improved interactive description with emojis, step-by-step flow and model names
st.markdown(
    """
### 🤖 How this model works (quick overview)

This app uses a **hierarchical two-model BERT system** to analyze news sentiment with improved precision.

1. **Model 1 — Sentiment Detector (Negative vs Not-Negative)** • Quickly detects whether the input expresses **negative** sentiment or **not-negative**.  
   • If it's **Negative**, the pipeline returns the result immediately.  

2. **Model 2 — Nuance Detector (Neutral vs Positive)** • Activated only when Model 1 predicts **Not-Negative**.  
   • Distinguishes between **Neutral** and **Positive** to capture subtle differences in news tone.

💡 **Why this helps:** splitting the task simplifies each model's job — the first finds negative signals, the second focuses on subtle positive/neutral distinctions. This reduces confusion and improves the positive-class performance.
"""
)

st.markdown(
    """
### ✨ What makes this special
- Trained on a **domain-specific dataset** with augmentation (synonym replacement & back-translation).  
- Designed to mimic human two-step reasoning (detect emotion → identify type).  
- Deployed for quick, interactive use via Streamlit.
"""
)

# --- USER INPUT ---
user_input = st.text_area(
    "📝 Enter a news headline or short article snippet:",
    placeholder="Example: 'The company reported record profits...' or 'A major data breach impacted customer accounts.'",
    height=150
)

# --- [THIS BLOCK WAS MOVED UP] ---
# Action button
if st.button("🔎 Analyze Sentiment"):
    if user_input.strip():
        # Perform prediction
        with st.spinner("Analyzing..."):
            result = predict_sentiment(user_input)
            sentiment = result['sentiment']
            confidence = result['confidence']
            all_probs = result['all_probs']
        
        # Display results with emoji + colored feedback
        if sentiment == "Positive":
            st.success(f"**Prediction: {sentiment}** (Confidence: {confidence * 100:.2f}%)")
        elif sentiment == "Negative":
            st.error(f"**Prediction: {sentiment}** (Confidence: {confidence * 100:.2f}%)")
        else:
            st.info(f"ℹ **Prediction: {sentiment}** (Confidence: {confidence * 100:.2f}%)")
        
        # Display the bar chart with emoji heading
        st.subheader("📊 Probability Distribution")
        df_probs = pd.DataFrame.from_dict(all_probs, orient="index", columns=["Probability"])
        st.bar_chart(df_probs)
        
        # Optional: show raw probabilities for transparency
        with st.expander("🧾 Show raw probabilities"):
            st.write(df_probs.T)
            
    else:
        st.warning("⚠️ Please enter some text to analyze.")

# --- [THIS BLOCK WAS MOVED DOWN] ---
# Add a small helper / model links area
with st.expander("🔍 About the models (click to expand)"):
    st.write("- **Model 1 (Sentiment Detector):** Negative vs Not-Negative")
    st.write("  - HF repo: https://huggingface.co/SourabhKhamankar/model-1-neg-vs-notneg")
    st.write("- **Model 2 (Nuance Detector):** Neutral vs Positive")
    st.write("  - HF repo: https://huggingface.co/SourabhKhamankar/model-2-neu-vs-pos")
    st.write("These models were fine-tuned for news sentiment and optimized for practical accuracy in production-like inputs.")


st.markdown("---")
