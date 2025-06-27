#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Raul Gomez
Date: 2025-03-19
Description: 
"""


import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

class SentimentModel:
    def __init__(self, num_words=1000, max_len=20, oov_token="<OOV>"):
        self.num_words = num_words
        self.max_len = max_len
        self.oov_token = oov_token
        self.tokenizer = Tokenizer(num_words=self.num_words, oov_token=self.oov_token)
        self.model = None

    def prepare_tokenizer(self, texts):
        self.tokenizer.fit_on_texts(texts)

    def texts_to_padded_sequences(self, texts):
        seqs = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(seqs, maxlen=self.max_len, padding='post')
        return padded

    def build_model(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(self.num_words, 16, input_length=self.max_len),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(3, activation='softmax')
        ])
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train(self, X_padded, y_onehot, epochs=100, verbose=0):
        self.build_model()
        self.model.fit(X_padded, y_onehot, epochs=epochs, verbose=verbose)

    def predict(self, texts):
        padded = self.texts_to_padded_sequences(texts)
        preds = self.model.predict(padded)
        return preds

    def save(self, model_path, tokenizer_path):
        if model_path.endswith(".keras"):
            self.model.save(model_path)
        else:
            self.model.save(model_path + ".keras")

        with open(tokenizer_path, 'wb') as f:
            pickle.dump(self.tokenizer, f)

    def load(self, model_path, tokenizer_path):
        self.model = tf.keras.models.load_model(model_path)
        with open(tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)
