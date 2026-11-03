import re

HEADER_RE = re.compile(r'(From:.*|Sent:.*|Subject:.*|To:.*|Cc:.*)', re.I)

def clean_email(t: str) -> str:
    t = t or ""
    # Cap to first 10k characters to prevent regex lag on massive HTML emails
    if len(t) > 10000:
        t = t[:10000]
    t = HEADER_RE.sub(' ', t)                         # drop quoted headers
    t = re.sub(r'https?://\S+|www\.\S+', ' URL ', t)  # URLs -> token
    t = re.sub(r'\b[\w\.-]+@[\w\.-]+\b', ' EMAIL ', t)# emails -> token
    t = re.sub(r'\d{4,}', ' NUMLONG ', t)             # long numbers
    t = re.sub(r'\d+', ' NUM ', t)                    # numbers
    t = re.sub(r'\s+', ' ', t).strip()
    return t

if __name__ == "__main__":
    sample_text = "From: sender@example.com\nSubject: Account Alert\n\nWinner! Check https://example.com for 123456 dollars."
    print("Original:\n", sample_text)
    print("\nCleaned:\n", clean_email(sample_text))
