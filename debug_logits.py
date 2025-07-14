#!/usr/bin/env python3
"""
Debug script to identify the 'logits' error in sentiment analysis model
"""

import sys
import os
sys.path.append('/home/hong/code/huggingface-gui')

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

def debug_sentiment_model():
    """Debug the sentiment analysis model to identify logits error"""
    
    model_path = "/home/hong/.cache/huggingface/hub/models--tabularisai--multilingual-sentiment-analysis/snapshots/69afb831cf544d133faa45fe320cfe42eba72376"
    
    print("=== Debugging Sentiment Analysis Model ===")
    print(f"Model path: {model_path}")
    
    try:
        # Method 1: Direct model loading
        print("\n1. Testing direct model loading...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        print(f"Model type: {type(model)}")
        print(f"Model config: {model.config}")
        print(f"Model labels: {model.config.label2id}")
        
        # Method 2: Test tokenization
        print("\n2. Testing tokenization...")
        test_text = "I love this product"
        inputs = tokenizer(test_text, return_tensors="pt")
        print(f"Tokenized inputs: {inputs}")
        
        # Method 3: Direct model inference
        print("\n3. Testing direct model inference...")
        with torch.no_grad():
            outputs = model(**inputs)
            print(f"Model outputs type: {type(outputs)}")
            print(f"Model outputs keys: {list(outputs.keys()) if hasattr(outputs, 'keys') else 'No keys'}")
            
            if hasattr(outputs, 'logits'):
                print(f"Logits shape: {outputs.logits.shape}")
                print(f"Logits: {outputs.logits}")
                
                # Apply softmax for probabilities
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                print(f"Probabilities: {probs}")
                
                # Get prediction
                predicted_class = torch.argmax(outputs.logits, dim=-1)
                print(f"Predicted class: {predicted_class}")
                
                # Map to label
                id2label = model.config.id2label
                predicted_label = id2label[predicted_class.item()]
                print(f"Predicted label: {predicted_label}")
        
        # Method 4: Test pipeline creation
        print("\n4. Testing pipeline creation...")
        try:
            # Test various pipeline creation methods
            pipe1 = pipeline("text-classification", model=model, tokenizer=tokenizer)
            print("Pipeline method 1 successful")
            
            result1 = pipe1(test_text)
            print(f"Pipeline result 1: {result1}")
            
        except Exception as e:
            print(f"Pipeline method 1 failed: {e}")
            print(f"Error type: {type(e)}")
            import traceback
            traceback.print_exc()
            
        try:
            # Test with model path
            pipe2 = pipeline("text-classification", model=model_path)
            print("Pipeline method 2 successful")
            
            result2 = pipe2(test_text)
            print(f"Pipeline result 2: {result2}")
            
        except Exception as e:
            print(f"Pipeline method 2 failed: {e}")
            print(f"Error type: {type(e)}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"Model loading failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_sentiment_model()