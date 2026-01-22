import streamlit as st
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="SMS Scam Detection System",
    page_icon="📩",
    layout="wide"
)

HF_MODEL_PATH = "Kisantini/SMS-Spam-Detection/sms_spam_distilbert_model"

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_PATH)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# ---------------- SIDEBAR ----------------
st.sidebar.title("📩 SMS Scam Detection")
st.sidebar.markdown("**NLP-based Classification System**")

menu = st.sidebar.radio(
    "Navigation",
    ["Single SMS Analysis", "Bulk SMS Analysis", "Analytics Dashboard", "About System"]
)

# ---------------- KEYWORDS ----------------
spam_keywords = [
    "free", "win", "prize", "urgent", "click", "offer",
    "limited", "congratulations", "claim", "reward"
]

# ---------------- PREDICTION FUNCTION ----------------
def predict_sms(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()

    label = "Spam" if pred == 1 else "Ham"
    return label, confidence, probs.squeeze().tolist()

# ---------------- SESSION HISTORY ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ================= SINGLE SMS =================
if menu == "Single SMS Analysis":
    st.title("🔍 Single SMS Analysis")
    st.write("Analyze an individual SMS message using the trained DistilBERT model.")

    text = st.text_area(
        "Enter SMS Message",
        height=150,
        placeholder="Example: Congratulations! You have won a free prize."
    )

    if st.button("Analyze"):
        if text.strip() == "":
            st.warning("Please enter a message.")
        else:
            label, confidence, probs = predict_sms(text)

            st.subheader("Prediction Result")

            if label == "Spam":
                st.error("🚨 Classified as SPAM")
            else:
                st.success("✅ Classified as HAM")

            # Confidence Bar
            st.write("### Prediction Confidence")
            st.progress(confidence)
            st.write(f"Confidence Score: **{confidence:.2f}**")

            # Explanation
            matched_keywords = [w for w in spam_keywords if w in text.lower()]
            st.write("### Explanation")
            if matched_keywords:
                st.write(
                    "The message contains suspicious keywords:",
                    ", ".join(matched_keywords)
                )
            else:
                st.write(
                    "The message does not contain common spam-related keywords."
                )

            # Save history
            st.session_state.history.append([text, label, round(confidence, 2)])

# ================= BULK SMS =================
elif menu == "Bulk SMS Analysis":
    st.title("📂 Bulk SMS Analysis")
    st.write("Upload a CSV file to classify multiple SMS messages.")

    st.info("CSV file must include a column named **message**")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)

        if "message" not in df.columns:
            st.error("CSV must contain a 'message' column.")
        else:
            st.dataframe(df.head())

            if st.button("Run Analysis"):
                preds = []
                confidences = []

                for msg in df["message"]:
                    label, conf, _ = predict_sms(str(msg))
                    preds.append(label)
                    confidences.append(conf)

                df["Prediction"] = preds
                df["Confidence"] = confidences

                st.subheader("Prediction Results")
                st.dataframe(df)

                # Download
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download Results",
                    csv,
                    "bulk_sms_predictions.csv",
                    "text/csv"
                )

# ================= ANALYTICS =================
elif menu == "Analytics Dashboard":
    st.title("📊 Analytics Dashboard")

    if len(st.session_state.history) == 0:
        st.info("No predictions available yet.")
    else:
        hist_df = pd.DataFrame(
            st.session_state.history,
            columns=["Message", "Prediction", "Confidence"]
        )

        st.subheader("Prediction Summary")
        st.dataframe(hist_df)

        # Count plot
        counts = hist_df["Prediction"].value_counts()

        fig, ax = plt.subplots()
        counts.plot(kind="bar", ax=ax)
        ax.set_title("Spam vs Ham Distribution")
        ax.set_ylabel("Count")
        ax.set_xlabel("Class")
        st.pyplot(fig)

# ================= ABOUT =================
else:
    st.title("ℹ️ About the System")

    st.markdown(f"""
    ### SMS Scam Detection Using NLP and Machine Learning

    This system was developed as part of an academic Natural Language Processing project.
    The objective is to automatically classify SMS messages as **Spam** or **Ham**.

    **Model Source**
    - `{HF_MODEL_PATH}`

    **Models Implemented**
    - Naive Bayes with TF-IDF (Baseline)
    - DistilBERT Transformer (Proposed)

    **Key Features**
    - Single and bulk SMS classification
    - Confidence-based predictions
    - Interactive analytics dashboard

    **Limitations**
    - Performance depends on dataset size
    - Transformer models require careful tuning for short-text data

    **Future Enhancements**
    - Scam type classification
    - Multilingual SMS support
    - Real-time SMS filtering
    """)

    st.write("Developed for academic demonstration purposes.")
