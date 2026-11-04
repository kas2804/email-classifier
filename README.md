# AI-Powered Email Classifier

A full-stack machine learning application that connects to Gmail and classifies inbox emails into **SPAM**, **FRAUD**, and **NORMAL** in real-time. Built with a Scikit-Learn TF-IDF + Logistic Regression pipeline, Streamlit frontend, and Google Gmail API integration.

---

## 📹 Demo Preview

<!-- Replace the link below with your demo video, GIF, or YouTube link -->


https://github.com/user-attachments/assets/484f87a8-a980-47d9-9998-e07fa6083398


> *(Click to watch the demo video)*

---

## ✨ Key Features

- **Multi-Class AI Classification**: Predicts whether an email is **NORMAL** (Safe), **SPAM** (Promotional/Junk), or **🚨 FRAUD** (Phishing/Scam).
- **Sub-Second Prediction**: In-memory cached inference runs in **~2 milliseconds**.
- **Live Gmail Sync**: Secure OAuth 2.0 integration with single-batch HTTP requests to load inbox previews in under 1 second.
- **Offline / Sample Mode**: Instant sample inbox with realistic phishing, spam, and normal emails for immediate demonstration.
- **Continuous Learning Feedback Loop**: Save user-corrected labels directly to `data/feedback.csv` to retrain and improve model accuracy.
- **Keyword Risk Detection**: Scans email subjects for high-risk phishing keywords (*urgent, wire transfer, verify account, winner*).

---

## 📁 Project Structure

```text
email-classifier/
├── .streamlit/
│   └── config.toml          # Streamlit server & file-watcher performance configuration
├── data/
│   ├── final_dataset.csv    # Training dataset
│   └── feedback.csv         # User feedback collection for active learning
├── models/
│   ├── classifier.joblib    # Trained TF-IDF + Logistic Regression model bundle
│   └── multilabel_lr.joblib # Auxiliary multi-label model
├── src/
│   ├── api.py               # Optional FastAPI REST endpoint
│   ├── clean.py             # Regex text cleaning & normalization
│   ├── gmail_list.py        # Native OAuth 2.0 & batch Gmail API client
│   └── train.py             # Model training & evaluation pipeline
├── ui/
│   └── app.py               # Streamlit interactive web application
├── credentials.example.json # Template for Google OAuth setup
├── requirements.txt         # Project dependencies
└── README.md
```

---

## 🚀 Quickstart Guide

### 1. Clone & Set Up Virtual Environment

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/email-classifier.git
cd email-classifier

# Create virtual environment
python3 -m venv .venv

# Activate environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

### 2. Run the Application

You can launch the full interactive app with a single command:

```bash
streamlit run ui/app.py
```

- **Instant Testing**: Click **"📬 Load Sample Inbox"** or open the **"✍️ Test Any Custom Email"** tab to test predictions immediately without any Gmail setup.
- **Live Gmail**: Follow the optional steps below to connect your personal inbox.

---

### 3. (Optional) Connect Your Live Gmail

This project uses the official Google Gmail API (read-only mode). To connect your real inbox:

1. Go to the [Google Cloud Console](https://console.cloud.google.com/) and create a project.
2. Enable the **Gmail API** under *APIs & Services ➔ Library*.
3. Under *OAuth consent screen*:
   - Set User Type to **External**.
   - Under **Test users**, add your own Gmail address.
4. Under *Credentials ➔ Create Credentials ➔ OAuth client ID*:
   - Choose **Desktop app**.
   - Download the JSON and save it as `credentials.json` in the project root directory.
5. In your terminal, run the one-time authentication:
   ```bash
   python src/gmail_list.py
   ```
6. Sign in through the browser prompt. Once complete, click **"📥 Fetch Live Gmail"** in the app!

> [!NOTE]
> `credentials.json` and `token.json` are excluded from version control via `.gitignore` to keep your credentials private.

---

## 🧠 Model Training

To retrain the model on the dataset or newly collected feedback:

```bash
python src/train.py
```

The script cleans the text, trains the TF-IDF vectorizer + Logistic Regression pipeline, prints a classification report, and saves the updated weights to `models/classifier.joblib`.

---

## 🔒 Security & Privacy

- `token.json` and `credentials.json` are strictly ignored by Git and never committed.
- Gmail access is **strictly read-only** (`gmail.readonly`); the application never sends, modifies, or deletes emails.
