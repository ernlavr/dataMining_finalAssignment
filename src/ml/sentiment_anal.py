from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
from torch.utils.data import DataLoader
import numpy as np
from scipy.special import softmax

class SentimentAnalysis:
    def __init__(self):
        self.model_name = "51la5/distilbert-base-sentiment"
        #self.model_name = f"cardiffnlp/twitter-xlm-roberta-base-sentiment"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.config = AutoConfig.from_pretrained(self.model_name)

        # PT
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.id2label = {0: "negative", 1: "neutral", 2: "positive"}
        self.label2id = {"negative": 0, "neutral": 1, "positive": 2}
    
        faux_tweets = [
            "I love you",
            "I hate you",
            "I don't care",
            "I don't know",
        ]
        # batch faux tweets in size 2

        dataloader = DataLoader(faux_tweets, batch_size=2)

        self.run_inference(faux_tweets)

    # Preprocess text (username and link placeholders)
    def preprocess(text):
        new_text = []
        for t in text.split(" "):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        return " ".join(new_text)

    def run_inference(self, tweets):
        text = self.preprocess(tweets)
        encoded_input = self.tokenizer.batch_encode(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        output = self.model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)

        # Print labels and scores
        ranking = np.argsort(scores)
        ranking = ranking[::-1]
        for i in range(scores.shape[0]):
            l = self.config.id2label[ranking[i]]
            s = scores[ranking[i]]
            print(f"{i+1}) {l} {np.round(float(s), 4)}")