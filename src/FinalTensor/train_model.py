#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Raul Gomez
Date: 2025-03-19
Description: 
"""


import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from FinalTensor.model import SentimentModel
import os

def main():
    # Dataset
    opiniones = [
        "Me encantó la clase, todo fue claro y bien explicado",
        "No entendí nada, fue una pérdida de tiempo",
        "Estuvo regular, algunas partes fueron interesantes",
        "Excelente docente, aprendí mucho",
        "Muy aburrido, no volvería a tomar esta materia",
        "No estuvo tan mal, aunque podría mejorar",
        "El curso fué demasiado rápido y no pude seguir el ritmo",
        "Las fechas de los quizes debieron estar mejor distribuidas",
        "La situación problema tenía poco que ver con lo que vimos en clase",
        "El profesor contesta todas las dudas y es muy paciente",
        "En realidad está todo bien",
        "Sus clases son decentes",
        "Puede mejorar el enfoque de sus clases",
        "Los métodos que utiliza lo hace parecer más complicado de lo que es",
        "Demuestra gran conocimiento de los temas",
        "No hace la clase dinámica",
        "Clases tediosas",
        "No brinda apoyo",
        "Me hubiera gustado entender un poco mejor los temas del examen final",
        "Que incentive más la participación de los alumnos",
        "Muestra un dominio de todos los temas que se vieron en clase"
    ]
    etiquetas = [
        'positivo',
        'negativo',
        'neutral',
        'positivo',
        'negativo',
        'neutral',
        'negativo',
        'negativo',
        'negativo',
        'positivo',
        'positivo',
        'neutral',
        'neutral',
        'negativo',
        'positivo',
        'negativo',
        'negativo',
        'negativo',
        'negativo',
        'neutral',
        'positivo'
    ]

    label_map = {'positivo': 0, 'negativo': 1, 'neutral': 2}
    y = to_categorical([label_map[e] for e in etiquetas], num_classes=3)

    # Initialize model
    max_len = 20  # or dynamic based on data
    model = SentimentModel(num_words=1000, max_len=max_len)

    # Prepare tokenizer and data
    model.prepare_tokenizer(opiniones)
    X_padded = model.texts_to_padded_sequences(opiniones)

    # Train
    model.train(X_padded, y, epochs=100, verbose=1)

    # Save model and tokenizer
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),"..", "..", "models"))
    os.makedirs(base_dir, exist_ok=True)
    model.save(
        os.path.join(base_dir, "sentiment_model.keras"),
        os.path.join(base_dir, "tokenizer.pkl")
    )
    print("Modelo entrenado y guardado exitosamente.")

if __name__ == "__main__":
    main()
