import re
import pandas as pd

# regex patterns for structured PII
pii_patterns = [
    r'\b(?:\d[ -]*?){13,19}\b',  # credit card number
    r'\b(?:\d{3}-\d{2}-\d{4}|\d{9})\b',  # social security number
    r'\b\d{10}\b',  # phone number
    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'  # email
]

def contains_structured_pii(text: str) -> bool:
    return any(re.search(pattern, text) for pattern in pii_patterns)


if __name__ == "__main__":

    df = pd.read_csv("/home/dsi/ohadico97/homework/data.csv")
    texts = df['text'].values
    targets = df['target'].values
    for idx,sen in enumerate(texts):
        print("Regex:",contains_structured_pii(sen),'Target:',targets[idx])

