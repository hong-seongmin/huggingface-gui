"""
빠른 대안 모델 제안 시스템
"""
import time
from typing import List, Dict, Any

class QuickAlternatives:
    """빠르게 로딩되는 대안 모델들을 제안하는 시스템"""
    
    def __init__(self):
        # 빠른 대안 모델들 (크기 순)
        self.fast_alternatives = {
            "sentiment": [
                {
                    "name": "cardiffnlp/twitter-roberta-base-sentiment-latest",
                    "size": "278MB",
                    "expected_load_time": "30-60초",
                    "description": "Twitter 감정 분석 (영어)",
                    "quality": "우수"
                },
                {
                    "name": "nlptown/bert-base-multilingual-uncased-sentiment",
                    "size": "714MB", 
                    "expected_load_time": "60-120초",
                    "description": "다국어 감정 분석",
                    "quality": "매우 우수"
                },
                {
                    "name": "distilbert-base-uncased-finetuned-sst-2-english",
                    "size": "255MB",
                    "expected_load_time": "20-45초", 
                    "description": "DistilBERT 감정 분석 (영어, 가장 빠름)",
                    "quality": "우수"
                }
            ],
            "general": [
                {
                    "name": "distilbert-base-uncased", 
                    "size": "255MB",
                    "expected_load_time": "20-45초",
                    "description": "DistilBERT 기본 모델 (가장 빠름)",
                    "quality": "우수"
                },
                {
                    "name": "bert-base-uncased",
                    "size": "440MB", 
                    "expected_load_time": "45-90초",
                    "description": "BERT 기본 모델",
                    "quality": "매우 우수"
                }
            ]
        }
    
    def suggest_fast_alternatives(self, current_model: str, task_type: str = "sentiment") -> List[Dict]:
        """빠른 대안 모델들 제안"""
        
        if task_type.lower() in ["sentiment", "classification"]:
            alternatives = self.fast_alternatives["sentiment"]
        else:
            alternatives = self.fast_alternatives["general"]
        
        # 현재 모델 크기 추정
        current_size = self._estimate_model_size(current_model)
        
        # 더 작은 모델들만 제안
        fast_alternatives = []
        for alt in alternatives:
            alt_size = self._parse_size(alt["size"])
            if alt_size < current_size:
                fast_alternatives.append(alt)
        
        return fast_alternatives
    
    def _estimate_model_size(self, model_name: str) -> float:
        """모델 크기 추정 (MB)"""
        # 알려진 모델들의 크기
        known_sizes = {
            "tabularisai/multilingual-sentiment-analysis": 517.0,
            "bert-base-multilingual": 714.0,
            "distilbert-base-uncased": 255.0,
            "bert-base-uncased": 440.0
        }
        
        return known_sizes.get(model_name, 500.0)  # 기본값 500MB
    
    def _parse_size(self, size_str: str) -> float:
        """크기 문자열을 MB로 변환"""
        size_str = size_str.upper().replace("MB", "").replace("GB", "")
        try:
            size = float(size_str.replace("GB", ""))
            if "GB" in size_str.upper():
                size *= 1024
            return size
        except:
            return 500.0
    
    def print_alternatives(self, current_model: str, task_type: str = "sentiment"):
        """빠른 대안들을 예쁘게 출력"""
        alternatives = self.suggest_fast_alternatives(current_model, task_type)
        
        if not alternatives:
            print("❌ 더 빠른 대안 모델을 찾을 수 없습니다.")
            return
        
        print("🚀 빠른 대안 모델 제안")
        print("=" * 60)
        print(f"현재 모델: {current_model}")
        print(f"현재 로딩 시간: 250-350초 (매우 느림)")
        print()
        print("⚡ 추천 대안 모델들:")
        print()
        
        for i, alt in enumerate(alternatives, 1):
            print(f"{i}. 🎯 {alt['name']}")
            print(f"   📦 크기: {alt['size']}")
            print(f"   ⏱️  예상 로딩: {alt['expected_load_time']}")
            print(f"   📝 설명: {alt['description']}")
            print(f"   ⭐ 품질: {alt['quality']}")
            print()
        
        print("💡 사용법:")
        print("   1. 위 모델 이름을 복사하세요")
        print("   2. 모델 경로 입력란에 붙여넣기하세요")
        print("   3. 로드 버튼을 클릭하세요")
        print()
        print("🎯 추천: distilbert 모델들이 가장 빠릅니다!")
        print("=" * 60)
    
    def performance_comparison(self):
        """성능 비교표 출력"""
        print("📊 모델 성능 비교표")
        print("=" * 80)
        print(f"{'모델명':<40} {'크기':<10} {'로딩시간':<15} {'품질':<10}")
        print("-" * 80)
        
        # 현재 모델
        print(f"{'tabularisai/multilingual-sentiment-analysis':<40} {'517MB':<10} {'250-350초':<15} {'매우우수':<10}")
        print(f"{'(현재 사용중 - 매우 느림)':<40} {'':<10} {'🐌':<15} {'⭐⭐⭐⭐⭐':<10}")
        print()
        
        # 빠른 대안들
        fast_models = [
            ("distilbert-base-uncased-finetuned-sst-2", "255MB", "20-45초", "우수", "⚡⚡⚡"),
            ("cardiffnlp/twitter-roberta-base-sentiment", "278MB", "30-60초", "우수", "⚡⚡"), 
            ("nlptown/bert-base-multilingual-uncased", "714MB", "60-120초", "매우우수", "⚡"),
        ]
        
        for model, size, time, quality, speed in fast_models:
            print(f"{model:<40} {size:<10} {time:<15} {quality:<10}")
            print(f"{'(추천 대안)':<40} {'':<10} {speed:<15} {'⭐⭐⭐⭐':<10}")
            print()
        
        print("=" * 80)
        print("🏆 결론: DistilBERT 계열 모델이 속도와 성능의 최적 균형점!")

# 전역 인스턴스
quick_alternatives = QuickAlternatives()

if __name__ == "__main__":
    # 테스트 실행
    quick_alternatives.print_alternatives("tabularisai/multilingual-sentiment-analysis", "sentiment")
    print()
    quick_alternatives.performance_comparison()