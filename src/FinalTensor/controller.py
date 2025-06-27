import numpy as np
from FinalTensor.model import SentimentModel

class SentimentController:
    def __init__(self, model_path, tokenizer_path):
        self.label_map = {0: 'positivo', 1: 'negativo', 2: 'neutral'}
        self.model = SentimentModel()
        self.model.load(model_path, tokenizer_path)

    def analyze_sentiment(self, text):
        preds = self.model.predict([text])[0]
        label_idx = np.argmax(preds)
        label = self.label_map[label_idx]
        return label
