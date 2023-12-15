from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
from torch.utils.data import DataLoader
import torch
import numpy as np
from scipy.special import softmax
from tqdm import tqdm

class SentimentAnalysis:
    def __init__(self):
        self.model_name = "51la5/distilbert-base-sentiment"
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        #self.model_name = f"cardiffnlp/twitter-xlm-roberta-base-sentiment"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.config = AutoConfig.from_pretrained(self.model_name)

        # PT
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.id2label = {0: "negative", 1: "neutral", 2: "positive"}
        self.label2id = {"negative": 0, "neutral": 1, "positive": 2}
        self.batch_size = 2
    
        faux_tweets = [
            "I love you",
            "I hate you",
            "I don't care",
            "I don't know",
        ]
        self.run_inference(faux_tweets)


    def run_inference(self, tweets):
        dataloader = DataLoader(tweets, batch_size=self.batch_size)
        output = []

        for batch in tqdm(dataloader):
            encoded_input = self.tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=512)
            inputs = {'input_ids': encoded_input['input_ids'].to(self.device),
                    'attention_mask': encoded_input['attention_mask'].to(self.device)}
            preds = self.model(**inputs)
            scores = preds.logits.detach().numpy()
            scores = softmax(scores)
            labels = np.argmax(scores, axis=1)
            
            for i, label in enumerate(labels):
                l = self.config.id2label[label]
                output.append(self.label2id[l.lower()])

        return output