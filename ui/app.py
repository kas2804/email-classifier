# ui/app.py
import streamlit as st
import requests
import os
import sys
import pandas as pd
from joblib import load

# Add project root to sys.path so modules in src/ can be cleanly imported
ROOT = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(ROOT)
if PARENT not in sys.path:
    sys.path.append(PARENT)

from src.clean import clean_email
from src.gmail_list import (
    gmail_service,
    list_message_ids,
    get_message_brief,
    batch_get_message_briefs,
    get_message_full,
    check_auth_status,
)

# --- Page Configuration ---
st.set_page_config(page_title="AI Email Classifier", layout="centered", page_icon="✉️")

# --- Custom CSS for Styling ---
def set_custom_style():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: radial-gradient(circle at center, #B9D9EB 0%, #F0F8FF 70%);
            background-attachment: fixed;
            background-size: cover;
        }

        /* Bordered containers */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border: 2px solid #1e88e5 !important;
            border-radius: 8px !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

set_custom_style()

# --- App Header ---
st.title("📧Email Classifier")
st.caption("Classify emails into SPAM, FRAUD, or NORMAL using Machine Learning")

# --- Constants ---
LABELS = ["SPAM", "FRAUD", "NORMAL"]
KEYWORD_BANK = [
    "urgent", "action required", "verify your account", "winner", "congratulations",
    "invoice", "payment", "receipt", "order", "shipping", "refund", "unusual sign-in",
    "security alert", "password reset", "investment", "opportunity", "inheritance",
    "wire transfer", "lottery", "claim your prize", "bank account", "limited time"
]

SAMPLE_EMAILS = [
    {
        "id": "sample-1",
        "subject": "Immediate Action Required: Unusual Sign-in Activity Detected on Your Account",
        "from": "security-alert@verifyacc.io",
        "snippet": "We have detected an unusual sign-in attempt from Russia. For your security, verify your identity immediately...",
        "body": "Dear Customer,\n\nWe have detected an unusual sign-in attempt on your account from an unrecognized device in a different country.\n\nLocation: Moscow, Russia\n\nFor your security, we have temporarily suspended your account. You must immediately verify your identity and confirm your recent activity.\n\nPlease click the link below to sign in: http://verifyacc.io\n\nFailure to do so within 12 hours will result in permanent suspension.\n\nThank you,\nSecurity Team"
    },
    {
        "id": "sample-2",
        "subject": "Hey kasturi - double your money without hard work",
        "from": "crypto-deal@fastmoney.com",
        "snippet": "I have a special deal for you. Do you wish to double your money without any hard work? Click the link below...",
        "body": "Hey kasturi,\n\nI have a special deal for you. Do you wish to double your money without any hardwork?\n\nClick the link below and fill in your bank account details:\nhttps://meet.google.com/tvm-hwzw-jph\n\nLimited slots available!"
    },
    {
        "id": "sample-3",
        "subject": "Congratulations! You won a $1,000 Walmart Gift Card",
        "from": "rewards@prize-promotions.net",
        "snippet": "Winner notice: Your email was drawn as today's winner! Claim your reward now before offer expires...",
        "body": "Congratulations!\n\nYou have been chosen as the lucky winner for a $1,000 Walmart shopping spree!\n\nClaim your reward now by clicking here: http://giftcard-rewards.net/claim?id=99281\n\nOffer valid for the next 24 hours only. Terms apply."
    },
    {
        "id": "sample-4",
        "subject": "Team Sync and Sprint Planning - Monday at 10 AM",
        "from": "alex.turner@company.com",
        "snippet": "Hi everyone, please review the sprint backlog before our planning meeting on Monday morning...",
        "body": "Hi everyone,\n\nHope you had a great weekend. Please review the sprint backlog and update Jira tickets ahead of our sync on Monday at 10:00 AM.\n\nMeeting link: https://meet.google.com/abc-defg-hij\n\nBest,\nAlex"
    },
    {
        "id": "sample-5",
        "subject": "Your receipt for order #48291 from Amazon.com",
        "from": "auto-confirm@amazon.com",
        "snippet": "Thanks for your order. We'll send a confirmation once your items have shipped...",
        "body": "Hi Kasturi,\n\nThank you for shopping with us. We thought you would like to know that your order #48291 has been confirmed.\n\nEstimated delivery date: Wednesday, Nov 4\nTotal Amount: $24.99\n\nView or track your order in your Amazon account."
    }
]

# --- Cached Resource: Models ---
@st.cache_resource
def load_classifier_model():
    """Loads the main multi-class classifier model bundle once into memory."""
    path = os.path.join(PARENT, "models", "classifier.joblib")
    if not os.path.exists(path):
        path = os.path.join("models", "classifier.joblib")
    return load(path)

# --- Cached Resource: Gmail API Service ---
@st.cache_resource
def get_cached_gmail_service():
    """Initializes and caches the Gmail API OAuth service client."""
    return gmail_service(open_browser=False)

# --- Cached Data: Fetching and Prediction Functions (TTL = 300s) ---
@st.cache_data(ttl=300)
def fetch_email_briefs(max_n: int = 10) -> list[dict]:
    """Fetches list of email previews from Gmail in a single batch request."""
    svc = get_cached_gmail_service()
    raw_msgs = list_message_ids(max_n, service=svc)
    ids = [m["id"] for m in raw_msgs]
    return batch_get_message_briefs(ids, service=svc)

@st.cache_data(ttl=300)
def fetch_email_detail(msg_id: str) -> dict:
    """Fetches full email details for a message ID."""
    for s in SAMPLE_EMAILS:
        if s["id"] == msg_id:
            return s
    svc = get_cached_gmail_service()
    return get_message_full(msg_id, service=svc)

def predict_email(subject: str, body: str) -> dict:
    """Predicts email category in ~2 milliseconds using the in-memory ML pipeline."""
    bundle = load_classifier_model()
    model = bundle["model"]
    labels = bundle["labels"]
    text = f"{subject or ''} {body or ''}".strip()
    if len(text) > 10000:
        text = text[:10000]
    cleaned_text = clean_email(text)

    prediction = model.predict([cleaned_text])[0]
    probabilities = model.predict_proba([cleaned_text])[0]
    proba_map = {label: float(prob) for label, prob in zip(labels, probabilities)}

    return {
        "prediction": prediction,
        "probabilities": proba_map,
    }

# --- State Management ---
if "briefs" not in st.session_state:
    st.session_state["briefs"] = []
if "current_email" not in st.session_state:
    st.session_state["current_email"] = None
if "predictions" not in st.session_state:
    st.session_state["predictions"] = {}

# --- Sidebar Controls ---
with st.sidebar:
    st.header("Settings")
    email_limit = st.slider("Max emails to fetch", min_value=5, max_value=25, value=10, step=5)

    st.divider()
    auth_ok, auth_msg = check_auth_status()
    st.caption("Gmail Connection Status:")
    if auth_ok:
        st.success("✅ Gmail: Connected & Ready")
    else:
        st.warning(f"⚠️ Gmail: {auth_msg}")
        st.caption("Run `python src/gmail_list.py` in terminal if not connected.")

# --- Main App Tabs ---
tab_inbox, tab_manual = st.tabs(["📧 Email Inbox", "Test Any Custom Email"])

# ==========================================================
# TAB 1: Email Inbox
# ==========================================================
with tab_inbox:
    col_live, col_demo, col_reset = st.columns([2, 2, 1])
    
    with col_live:
        if st.button("📥 Fetch Live Gmail", use_container_width=True, type="primary"):
            try:
                with st.spinner("Connecting to your Gmail and fetching emails..."):
                    briefs = fetch_email_briefs(email_limit)
                    st.session_state["briefs"] = briefs
                    st.session_state["current_email"] = None
                    st.success(f"Fetched {len(briefs)} emails from your Gmail!")
                    st.rerun()
            except Exception as e:
                st.error(f"Gmail Error: {e}")
                st.info("💡 Make sure `token.json` is generated by running `python src/gmail_list.py` in your terminal.")

    with col_demo:
        if st.button("Load Sample Inbox", use_container_width=True):
            st.session_state["briefs"] = SAMPLE_EMAILS
            st.session_state["current_email"] = None
            st.rerun()

    with col_reset:
        if st.button("🔄 Clear", use_container_width=True):
            fetch_email_briefs.clear()
            fetch_email_detail.clear()
            st.session_state["briefs"] = []
            st.session_state["current_email"] = None
            st.rerun()

    # Email List Display
    if st.session_state["briefs"]:
        st.markdown(f"### Showing {len(st.session_state['briefs'])} Emails")
        for b in st.session_state["briefs"]:
            with st.container(border=True):
                c1, c2 = st.columns([4, 1])
                with c1:
                    st.markdown(f"**Subject:** {b.get('subject', '(no subject)')}")
                    st.caption(f"**From:** {b.get('from', '(unknown)')}")
                with c2:
                    if st.button("Open", key=f"open-{b['id']}", use_container_width=True):
                        with st.spinner("Opening..."):
                            st.session_state["current_email"] = fetch_email_detail(b["id"])
                            st.rerun()

    # Opened Email View & Classifier
    if st.session_state["current_email"]:
        cur = st.session_state["current_email"]
        email_id = cur.get("id", "")
        st.divider()
        st.subheader(f"📖 {cur.get('subject', '(no subject)')}")

        subject_lower = (cur.get("subject") or "").lower()
        found_keywords = [k for k in KEYWORD_BANK if k in subject_lower]
        if found_keywords:
            st.info(f"💡 Keywords detected in subject: {', '.join(found_keywords)}")

        email_body = cur.get("body") or "(no body content)"
        st.text_area("Email Body", email_body[:3000], height=180, disabled=True)

        if st.button("🔍 Classify This Email", type="primary", key="classify-inbox-btn"):
            out = predict_email(cur.get("subject") or "", cur.get("body") or "")
            st.session_state["predictions"][email_id] = out
            st.rerun()

        # Display result
        if email_id in st.session_state["predictions"]:
            res = st.session_state["predictions"][email_id]
            pred = res.get("prediction", "UNKNOWN")
            probs = res.get("probabilities", {})

            if pred == "NORMAL":
                st.success("### Prediction: ✅ NORMAL (Safe Email)")
            elif pred == "SPAM":
                st.warning("### Prediction: ⚠️ SPAM (Promotional/Junk)")
            elif pred == "FRAUD":
                st.error("### Prediction: 🚨 FRAUD (Phishing / Scam Alert)")

            if probs:
                p_cols = st.columns(len(probs))
                for i, (lbl, val) in enumerate(probs.items()):
                    with p_cols[i]:
                        st.metric(lbl, f"{val * 100:.1f}%")

            # Feedback
            st.write("---")
            fb_key = f"fb-{email_id}"
            is_ok = st.radio("Is this prediction correct?", ["Yes", "No"], key=fb_key, horizontal=True)
            if is_ok == "No":
                corr = st.selectbox("Select correct category:", LABELS, key=f"corr-{email_id}")
                if st.button("Save Feedback", key=f"save-fb-{email_id}"):
                    fb_path = os.path.join(PARENT, "data", "feedback.csv")
                    row = {
                        "Email": (cur.get("subject") or "") + " " + (cur.get("body") or ""),
                        "Label": corr,
                        "Length": len((cur.get("subject") or "") + " " + (cur.get("body") or ""))
                    }
                    os.makedirs(os.path.dirname(fb_path), exist_ok=True)
                    pd.DataFrame([row]).to_csv(fb_path, mode='a', header=not os.path.exists(fb_path), index=False)
                    st.success("Feedback recorded to data/feedback.csv!")

# ==========================================================
# TAB 2: Instant Manual Tester
# ==========================================================
with tab_manual:
    st.markdown("Test the AI classifier immediately on any custom or copied email message:")
    
    preset = st.selectbox(
        "Load a preset template:",
        [
            "(Custom)",
            "Scam / Fraud: Inheritance Bank Transfer",
            "Spam: Urgent Winner Notification",
            "Normal: Project Status Meeting Tomorrow"
        ]
    )

    default_subj = ""
    default_body = ""
    if preset == "Scam / Fraud: Inheritance Bank Transfer":
        default_subj = "URGENT: Your inheritance fund of $5,500,000 USD is pending approval"
        default_body = "Dear beneficiary, verify your bank account and passport number immediately to release your inheritance transfer. Action required within 24 hours."
    elif preset == "Spam: Urgent Winner Notification":
        default_subj = "Congratulations! You are selected as our winner"
        default_body = "Claim your $500 gift card now! Limited time offer. Click the URL below to receive your special discount coupon and reward today."
    elif preset == "Normal: Project Status Meeting Tomorrow":
        default_subj = "Weekly Project Update & Sync"
        default_body = "Hi team, please find attached the weekly engineering progress report. Let's sync tomorrow at 10 AM to discuss next milestones. Best regards."

    test_subject = st.text_input("Subject:", value=default_subj, placeholder="e.g. Action Required: Account Update")
    test_body = st.text_area("Body:", value=default_body, height=140, placeholder="Paste or type email text here...")

    if st.button("⚡ Classify Instantly", type="primary", key="classify-manual-btn"):
        if not test_subject.strip() and not test_body.strip():
            st.warning("Please enter a subject or body to classify.")
        else:
            out = predict_email(test_subject, test_body)
            st.session_state["manual_result"] = out

    if "manual_result" in st.session_state:
        out = st.session_state["manual_result"]
        pred = out.get("prediction", "UNKNOWN")
        probs = out.get("probabilities", {})

        if pred == "NORMAL":
            st.success(f"### Result: ✅ NORMAL")
        elif pred == "SPAM":
            st.warning(f"### Result: ⚠️ SPAM")
        elif pred == "FRAUD":
            st.error(f"### Result: 🚨 FRAUD")

        if probs:
            p_cols = st.columns(len(probs))
            for i, (lbl, val) in enumerate(probs.items()):
                with p_cols[i]:
                    st.metric(lbl, f"{val * 100:.1f}%")

st.divider()
st.caption("AI Email Classifier • Powered by Scikit-Learn TF-IDF + Logistic Regression")
