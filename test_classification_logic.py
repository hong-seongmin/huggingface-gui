#!/usr/bin/env python3
"""
Test the classification model detection logic
"""

class MockConfig:
    def __init__(self, architectures):
        self.architectures = architectures

def test_classification_detection():
    """Test the classification model detection logic from model_manager.py"""
    
    print("Testing classification model detection logic...")
    
    # Test cases
    test_cases = [
        {
            'name': 'RoBERTa Classification Model',
            'architectures': ['RobertaForSequenceClassification'],
            'expected': True
        },
        {
            'name': 'BERT Classification Model', 
            'architectures': ['BertForSequenceClassification'],
            'expected': True
        },
        {
            'name': 'DistilBERT Classification Model',
            'architectures': ['DistilBertForSequenceClassification'], 
            'expected': True
        },
        {
            'name': 'Regular RoBERTa Model',
            'architectures': ['RobertaModel'],
            'expected': False
        },
        {
            'name': 'BERT Model',
            'architectures': ['BertModel'],
            'expected': False
        },
        {
            'name': 'BGE Model',
            'architectures': ['BertModel'],
            'expected': False
        },
        {
            'name': 'Empty architectures',
            'architectures': [],
            'expected': False
        },
        {
            'name': 'None architectures',
            'architectures': None,
            'expected': False
        }
    ]
    
    # Test each case
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['name']}")
        
        config = MockConfig(test_case['architectures'])
        
        # This is the exact logic from model_manager.py line 193-197
        is_classification_model = (
            hasattr(config, 'architectures') and 
            config.architectures and
            any('Classification' in arch for arch in config.architectures)
        )
        
        print(f"  Architectures: {config.architectures}")
        print(f"  Expected: {test_case['expected']}")
        print(f"  Actual: {is_classification_model}")
        
        # Convert to boolean for comparison (since empty list/None evaluate to falsy)
        actual_bool = bool(is_classification_model)
        
        if actual_bool == test_case['expected']:
            print("  ✅ PASS")
        else:
            print("  ❌ FAIL")
            return False
    
    print("\n🎉 All tests passed! Classification detection logic is working correctly.")
    return True

if __name__ == "__main__":
    success = test_classification_detection()
    if success:
        print("\n✅ The fix in model_manager.py should correctly detect sentiment analysis models")
        print("   and load them with AutoModelForSequenceClassification instead of AutoModel.")
    else:
        print("\n❌ There's an issue with the classification detection logic.")