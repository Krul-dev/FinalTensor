#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Raul Gomez
Date: 2025-03-19
Description: 
"""

import streamlit as st
from FinalTensor.controller import SentimentController
import os

@st.cache_resource
def load_controller():
    model_path = os.path.join(os.path.dirname(__file__), "../models/sentiment_model.keras")
    tokenizer_path = os.path.join(os.path.dirname(__file__), "../models/tokenizer.pkl")
    return SentimentController(model_path, tokenizer_path)

def main():
    st.title("Análisis de los Comentarios de los Estudiantes en las ECOAS")

    controller = load_controller()

    user_input = st.text_input("Escribe el comentario recibido:")

    if st.button("Analizar el comentario"):
        if user_input.strip():
            result = controller.analyze_sentiment(user_input)
            st.success(f"El comentario recibido es: {result}")
        else:
            st.warning("Por favor ingresa un texto.")

if __name__ == "__main__":
    main()
