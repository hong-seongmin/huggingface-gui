#!/usr/bin/env python3
"""
Fix the broken load_model_async function
"""

load_model_async_code = '''    def load_model_async(self, model_name: str, model_path: str, callback: Optional[Callable] = None):
        """비동기 모델 로드 (HuggingFace 모델 ID 지원, 모델 이름 자동 생성)"""
        # 모델 이름이 비어있으면 자동 생성
        if not model_name or not model_name.strip():
            model_name = self._generate_model_name(model_path)
        
        # 중복 모델 이름 처리
        original_name = model_name
        counter = 1
        while model_name in self.models:
            model_name = f"{original_name}_{counter}"
            counter += 1
        
        print(f"[DEBUG] load_model_async 시작: {model_name}, {model_path}")
        
        # 로딩 락 설정
        if model_name not in self.loading_locks:
            self.loading_locks[model_name] = threading.Lock()
        
        # 스레드 시작
        print(f"[DEBUG] 스레드 생성 중: {model_name}")
        thread = threading.Thread(
            target=self._load_model_sync, 
            args=(model_name, model_path, callback),
            name=f"ModelLoad-{model_name}"
        )
        thread.daemon = True
        print(f"[DEBUG] 스레드 시작 전: {model_name}")
        thread.start()
        print(f"[DEBUG] 스레드 시작됨: {model_name}, thread={thread}")
        
        return thread

    def _load_model_sync(self, model_name: str, model_path: str, callback: Optional[Callable] = None):
        """실제 모델 로딩 작업 (스레드에서 실행)"""
        try:
            print(f"[DEBUG] _load_model_sync 시작: {model_name}, {model_path}")
            
            # 모델 정보 초기화
            self.models[model_name] = ModelInfo(
                name=model_name, 
                path=model_path, 
                status="loading"
            )
            
            print(f"[DEBUG] 모델 정보 초기화됨: {model_name}")
            self._notify_callbacks(model_name, "loading_started", {})
            
            # 메모리 사용량 측정 시작
            import psutil
            process = psutil.Process()
            mem_before = process.memory_info().rss
            
            # HuggingFace 모델 ID인지 확인하고 다운로드
            actual_model_path = model_path
            if self._is_huggingface_model_id(model_path):
                self._notify_callbacks(model_name, "downloading", {'model_id': model_path})
                actual_model_path = self._download_huggingface_model(model_path)
                self.models[model_name].path = actual_model_path  # 실제 경로로 업데이트
            
            # 모델 분석
            analysis = self.model_analyzer.analyze_model_directory(actual_model_path)
            self.models[model_name].config_analysis = analysis
            
            # 모델 로드
            from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
            
            # 디바이스 설정
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # accelerate 사용 가능 여부 확인
            try:
                import accelerate
                use_device_map = device == "cuda"
            except ImportError:
                use_device_map = False
            
            # 설정에서 architecture 확인
            config = AutoConfig.from_pretrained(actual_model_path)
            is_classification_model = (
                hasattr(config, 'architectures') and 
                config.architectures and
                any('Classification' in arch for arch in config.architectures)
            )
            
            print(f"[DEBUG] 모델 로딩 시작: classification={is_classification_model}, device_map={use_device_map}")
            
            if use_device_map:
                if is_classification_model:
                    model = AutoModelForSequenceClassification.from_pretrained(
                        actual_model_path,
                        torch_dtype=torch.float16,
                        device_map="auto"
                    )
                else:
                    model = AutoModel.from_pretrained(
                        actual_model_path,
                        torch_dtype=torch.float16,
                        device_map="auto"
                    )
            else:
                if is_classification_model:
                    model = AutoModelForSequenceClassification.from_pretrained(
                        actual_model_path,
                        torch_dtype=torch.float16 if device == "cuda" else torch.float32
                    )
                else:
                    model = AutoModel.from_pretrained(
                        actual_model_path,
                        torch_dtype=torch.float16 if device == "cuda" else torch.float32
                    )
                # accelerate 없으면 수동으로 디바이스 설정
                if device == "cuda":
                    model = model.to(device)
            
            tokenizer = AutoTokenizer.from_pretrained(actual_model_path)
            
            # 메모리 사용량 계산
            mem_after = process.memory_info().rss
            memory_usage = (mem_after - mem_before) / 1024 / 1024  # MB
            
            # 모델 정보 업데이트
            self.models[model_name].model = model
            self.models[model_name].tokenizer = tokenizer
            self.models[model_name].memory_usage = memory_usage
            self.models[model_name].load_time = datetime.now()
            self.models[model_name].status = "loaded"
            
            success_data = {
                'memory_usage': memory_usage,
                'load_time': self.models[model_name].load_time,
                'analysis': analysis['model_summary'],
                'original_path': model_path,
                'actual_path': actual_model_path
            }
            
            self._notify_callbacks(model_name, "loading_success", success_data)
            
            if callback:
                callback(model_name, True, f"Model loaded successfully. Memory usage: {memory_usage:.2f} MB")
                
        except Exception as e:
            error_msg = str(e)
            print(f"[DEBUG] 모델 로딩 오류: {error_msg}")
            import traceback
            traceback.print_exc()
            
            if model_name in self.models:
                self.models[model_name].status = "error"
                self.models[model_name].error_message = error_msg
            
            self._notify_callbacks(model_name, "loading_error", {'error': error_msg})
            
            if callback:
                callback(model_name, False, error_msg)
'''

print("Fixed load_model_async function:")
print(load_model_async_code)