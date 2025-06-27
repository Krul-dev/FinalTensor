#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Raul Gomez
Date: 2025-03-19
Description: 
"""


import subprocess
import os
from train_model import main as train_model_main

def git_add_and_commit(file_paths, message):
    for path in file_paths:
        subprocess.run(["git", "add", path], check=True)
    subprocess.run(["git", "commit", "-m", message], check=True)

def git_push():
    subprocess.run(["git", "push"], check=True)

def main():
    # Step 1: Train and save model
    train_model_main()

    # Step 2: Git add + commit + push
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models", "sentiment_model.keras"))
    tokenizer_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models", "tokenizer.pkl"))

    try:
        git_add_and_commit([model_path, tokenizer_path], "Update trained sentiment model")
        print("✅ Model committed to Git.")
    except subprocess.CalledProcessError:
        print("⚠️ Could not commit files. Are there any changes?")
    
    try:
        git_push()
        print("✅ Pushed to GitHub.")
    except subprocess.CalledProcessError:
        print("⚠️ Push failed. Are you on the correct branch? Authenticated?")

if __name__ == "__main__":
    main()
