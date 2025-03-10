import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from distilbert import create_train

# set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

MAX_LEN = 64
BATCH_SIZE = 16
MODEL_NAME = 'distilbert-base-uncased'  # smaller and faster than full BERT
RANDOM_SEED = 42
CSV_PATH = "/home/dsi/ohadico97/homework/data.csv"
LOAD_TRAINED_MODEL = "/home/dsi/ohadico97/homework/pii_model.pt"


torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)



def detect_pii(texts: list[str]) -> list[bool]:
    # check if model exists, otherwise create and train
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=2
        )
        model.load_state_dict(torch.load(LOAD_TRAINED_MODEL, map_location=device))
        model = model.to(device)
    except:
        model, tokenizer = create_train(CSV_PATH)
    
    # set model to evaluation mode
    model.eval()
    

    results = []
    batch_size = 3  
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]

        encodings = tokenizer(
            batch_texts,
            add_special_tokens=False,
            max_length=MAX_LEN,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs.logits, dim=1)
            
            batch_results = [bool(pred) for pred in preds.cpu().tolist()]
            results.extend(batch_results)
    
    return results

if __name__ == "__main__":
    texts = [
        "Aunt Petunia's credit card number is 1234-5678-1234-5678.",# True
        "My uncle locked me in the cupboard under the stairs.", # False
        "Where is my cat?" # Added by me - False
    ]
    detections = detect_pii(texts)
    print(detections)