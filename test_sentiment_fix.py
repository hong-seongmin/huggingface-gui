#!/usr/bin/env python3
"""
Test script to verify sentiment analysis model fix
"""

def test_sentiment_model_fix():
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
        import torch
        
        # Test a known sentiment analysis model
        model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        print(f"Testing sentiment analysis model: {model_path}")
        
        # Check config first
        print("1. Checking model config...")
        config = AutoConfig.from_pretrained(model_path)
        print(f"   Model architectures: {config.architectures}")
        
        # Determine if it's a classification model
        is_classification_model = (
            hasattr(config, 'architectures') and 
            config.architectures and
            any('Classification' in arch for arch in config.architectures)
        )
        print(f"   Is classification model: {is_classification_model}")
        
        # Load model with appropriate class
        print("2. Loading model...")
        if is_classification_model:
            print("   Using AutoModelForSequenceClassification")
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
        else:
            print("   Using AutoModel")
            from transformers import AutoModel
            model = AutoModel.from_pretrained(model_path)
        
        # Load tokenizer
        print("3. Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Test inference
        print("4. Testing inference...")
        test_text = "I love this product! It's amazing."
        inputs = tokenizer(test_text, return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
            print(f"   Model output keys: {outputs.keys()}")
            
            if hasattr(outputs, 'logits'):
                print(f"   Logits shape: {outputs.logits.shape}")
                
                # Test classification prediction
                import torch.nn.functional as F
                probs = F.softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(outputs.logits, dim=-1)
                
                # Get label mapping
                if hasattr(model.config, 'id2label'):
                    id2label = model.config.id2label
                    predicted_label = id2label.get(predicted_class.item(), f"LABEL_{predicted_class.item()}")
                    print(f"   Predicted class: {predicted_class.item()}")
                    print(f"   Predicted label: {predicted_label}")
                    print(f"   Confidence: {probs[0][predicted_class.item()].item():.4f}")
                    
                    # Show all probabilities
                    print("   All predictions:")
                    for i, (label_id, label_name) in enumerate(id2label.items()):
                        prob = probs[0][i].item()
                        print(f"     {label_name}: {prob:.4f}")
                else:
                    print(f"   Raw prediction: class {predicted_class.item()}")
                
                print("✅ SUCCESS: Model loaded and inference working correctly!")
                return True
            else:
                print(f"❌ ERROR: Model output does not contain logits: {outputs}")
                return False
                
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_sentiment_model_fix()
    print(f"\nTest result: {'PASSED' if success else 'FAILED'}")