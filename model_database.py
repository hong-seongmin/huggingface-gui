"""
Hugging Face 모델 타입 데이터베이스
공식 transformers 라이브러리의 모델 타입 정보를 체계적으로 관리
"""

import re
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum

class ModelCategory(Enum):
    """모델 카테고리 분류"""
    TEXT = "text"
    VISION = "vision"
    AUDIO = "audio"
    MULTIMODAL = "multimodal"
    TIME_SERIES = "time_series"
    GRAPH = "graph"
    SPECIAL = "special"

class TaskType(Enum):
    """태스크 유형 분류"""
    TEXT_CLASSIFICATION = "text-classification"
    TOKEN_CLASSIFICATION = "token-classification"
    QUESTION_ANSWERING = "question-answering"
    TEXT_GENERATION = "text-generation"
    TEXT2TEXT_GENERATION = "text2text-generation"
    FEATURE_EXTRACTION = "feature-extraction"
    FILL_MASK = "fill-mask"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    IMAGE_CLASSIFICATION = "image-classification"
    OBJECT_DETECTION = "object-detection"
    IMAGE_SEGMENTATION = "image-segmentation"
    AUTOMATIC_SPEECH_RECOGNITION = "automatic-speech-recognition"
    AUDIO_CLASSIFICATION = "audio-classification"
    TEXT_TO_SPEECH = "text-to-speech"
    ZERO_SHOT_CLASSIFICATION = "zero-shot-classification"
    SENTENCE_SIMILARITY = "sentence-similarity"

@dataclass
class ModelTypeInfo:
    """모델 타입 정보"""
    model_type: str
    config_class: str
    category: ModelCategory
    primary_tasks: List[TaskType]
    description: str
    doc_link: str
    
class HuggingFaceModelDatabase:
    """Hugging Face 모델 타입 데이터베이스"""
    
    def __init__(self):
        self.models = self._build_model_database()
        
    def _build_model_database(self) -> Dict[str, ModelTypeInfo]:
        """모델 데이터베이스 구축"""
        models = {}
        
        # HTML 리스트에서 추출한 모델 정보
        model_data = [
            # 텍스트 모델들
            ("albert", "AlbertConfig", ModelCategory.TEXT, [TaskType.TEXT_CLASSIFICATION, TaskType.TOKEN_CLASSIFICATION, TaskType.FILL_MASK], "ALBERT model", "/docs/transformers/v4.53.2/en/model_doc/albert#transformers.AlbertConfig"),
            ("bert", "BertConfig", ModelCategory.TEXT, [TaskType.TEXT_CLASSIFICATION, TaskType.TOKEN_CLASSIFICATION, TaskType.FILL_MASK], "BERT model", "/docs/transformers/v4.53.2/en/model_doc/bert#transformers.BertConfig"),
            ("distilbert", "DistilBertConfig", ModelCategory.TEXT, [TaskType.TEXT_CLASSIFICATION, TaskType.TOKEN_CLASSIFICATION, TaskType.FILL_MASK], "DistilBERT model", "/docs/transformers/v4.53.2/en/model_doc/distilbert#transformers.DistilBertConfig"),
            ("roberta", "RobertaConfig", ModelCategory.TEXT, [TaskType.TEXT_CLASSIFICATION, TaskType.TOKEN_CLASSIFICATION, TaskType.FILL_MASK], "RoBERTa model", "/docs/transformers/v4.53.2/en/model_doc/roberta#transformers.RobertaConfig"),
            ("xlm-roberta", "XLMRobertaConfig", ModelCategory.TEXT, [TaskType.TEXT_CLASSIFICATION, TaskType.TOKEN_CLASSIFICATION, TaskType.FILL_MASK], "XLM-RoBERTa model", "/docs/transformers/v4.53.2/en/model_doc/xlm-roberta#transformers.XLMRobertaConfig"),
            ("deberta", "DebertaConfig", ModelCategory.TEXT, [TaskType.TEXT_CLASSIFICATION, TaskType.TOKEN_CLASSIFICATION], "DeBERTa model", "/docs/transformers/v4.53.2/en/model_doc/deberta#transformers.DebertaConfig"),
            ("deberta-v2", "DebertaV2Config", ModelCategory.TEXT, [TaskType.TEXT_CLASSIFICATION, TaskType.TOKEN_CLASSIFICATION], "DeBERTa-v2 model", "/docs/transformers/v4.53.2/en/model_doc/deberta-v2#transformers.DebertaV2Config"),
            ("electra", "ElectraConfig", ModelCategory.TEXT, [TaskType.TEXT_CLASSIFICATION, TaskType.TOKEN_CLASSIFICATION, TaskType.FILL_MASK], "ELECTRA model", "/docs/transformers/v4.53.2/en/model_doc/electra#transformers.ElectraConfig"),
            ("camembert", "CamembertConfig", ModelCategory.TEXT, [TaskType.TEXT_CLASSIFICATION, TaskType.TOKEN_CLASSIFICATION, TaskType.FILL_MASK], "CamemBERT model", "/docs/transformers/v4.53.2/en/model_doc/camembert#transformers.CamembertConfig"),
            ("xlnet", "XLNetConfig", ModelCategory.TEXT, [TaskType.TEXT_CLASSIFICATION, TaskType.TOKEN_CLASSIFICATION, TaskType.QUESTION_ANSWERING], "XLNet model", "/docs/transformers/v4.53.2/en/model_doc/xlnet#transformers.XLNetConfig"),
            ("longformer", "LongformerConfig", ModelCategory.TEXT, [TaskType.TEXT_CLASSIFICATION, TaskType.TOKEN_CLASSIFICATION, TaskType.QUESTION_ANSWERING], "Longformer model", "/docs/transformers/v4.53.2/en/model_doc/longformer#transformers.LongformerConfig"),
            ("big_bird", "BigBirdConfig", ModelCategory.TEXT, [TaskType.TEXT_CLASSIFICATION, TaskType.TOKEN_CLASSIFICATION, TaskType.QUESTION_ANSWERING], "BigBird model", "/docs/transformers/v4.53.2/en/model_doc/big_bird#transformers.BigBirdConfig"),
            ("convbert", "ConvBertConfig", ModelCategory.TEXT, [TaskType.TEXT_CLASSIFICATION, TaskType.TOKEN_CLASSIFICATION], "ConvBERT model", "/docs/transformers/v4.53.2/en/model_doc/convbert#transformers.ConvBertConfig"),
            ("mobilebert", "MobileBertConfig", ModelCategory.TEXT, [TaskType.TEXT_CLASSIFICATION, TaskType.TOKEN_CLASSIFICATION, TaskType.QUESTION_ANSWERING], "MobileBERT model", "/docs/transformers/v4.53.2/en/model_doc/mobilebert#transformers.MobileBertConfig"),
            ("squeezebert", "SqueezeBertConfig", ModelCategory.TEXT, [TaskType.TEXT_CLASSIFICATION, TaskType.TOKEN_CLASSIFICATION], "SqueezeBERT model", "/docs/transformers/v4.53.2/en/model_doc/squeezebert#transformers.SqueezeBertConfig"),
            
            # 생성 모델들
            ("gpt2", "GPT2Config", ModelCategory.TEXT, [TaskType.TEXT_GENERATION], "OpenAI GPT-2 model", "/docs/transformers/v4.53.2/en/model_doc/gpt2#transformers.GPT2Config"),
            ("gpt_neo", "GPTNeoConfig", ModelCategory.TEXT, [TaskType.TEXT_GENERATION], "GPT Neo model", "/docs/transformers/v4.53.2/en/model_doc/gpt_neo#transformers.GPTNeoConfig"),
            ("gpt_neox", "GPTNeoXConfig", ModelCategory.TEXT, [TaskType.TEXT_GENERATION], "GPT NeoX model", "/docs/transformers/v4.53.2/en/model_doc/gpt_neox#transformers.GPTNeoXConfig"),
            ("gptj", "GPTJConfig", ModelCategory.TEXT, [TaskType.TEXT_GENERATION], "GPT-J model", "/docs/transformers/v4.53.2/en/model_doc/gptj#transformers.GPTJConfig"),
            ("gpt_bigcode", "GPTBigCodeConfig", ModelCategory.TEXT, [TaskType.TEXT_GENERATION], "GPTBigCode model", "/docs/transformers/v4.53.2/en/model_doc/gpt_bigcode#transformers.GPTBigCodeConfig"),
            ("llama", "LlamaConfig", ModelCategory.TEXT, [TaskType.TEXT_GENERATION], "LLaMA model", "/docs/transformers/v4.53.2/en/model_doc/llama#transformers.LlamaConfig"),
            ("llama4", "Llama4Config", ModelCategory.TEXT, [TaskType.TEXT_GENERATION], "Llama4 model", "/docs/transformers/v4.53.2/en/model_doc/llama4#transformers.Llama4Config"),
            ("mistral", "MistralConfig", ModelCategory.TEXT, [TaskType.TEXT_GENERATION], "Mistral model", "/docs/transformers/v4.53.2/en/model_doc/mistral#transformers.MistralConfig"),
            ("mistral3", "Mistral3Config", ModelCategory.TEXT, [TaskType.TEXT_GENERATION], "Mistral3 model", "/docs/transformers/v4.53.2/en/model_doc/mistral3#transformers.Mistral3Config"),
            ("mixtral", "MixtralConfig", ModelCategory.TEXT, [TaskType.TEXT_GENERATION], "Mixtral model", "/docs/transformers/v4.53.2/en/model_doc/mixtral#transformers.MixtralConfig"),
            ("bloom", "BloomConfig", ModelCategory.TEXT, [TaskType.TEXT_GENERATION], "BLOOM model", "/docs/transformers/v4.53.2/en/model_doc/bloom#transformers.BloomConfig"),
            ("opt", "OPTConfig", ModelCategory.TEXT, [TaskType.TEXT_GENERATION], "OPT model", "/docs/transformers/v4.53.2/en/model_doc/opt#transformers.OPTConfig"),
            ("falcon", "FalconConfig", ModelCategory.TEXT, [TaskType.TEXT_GENERATION], "Falcon model", "/docs/transformers/v4.53.2/en/model_doc/falcon#transformers.FalconConfig"),
            ("gemma", "GemmaConfig", ModelCategory.TEXT, [TaskType.TEXT_GENERATION], "Gemma model", "/docs/transformers/v4.53.2/en/model_doc/gemma#transformers.GemmaConfig"),
            ("gemma2", "Gemma2Config", ModelCategory.TEXT, [TaskType.TEXT_GENERATION], "Gemma2 model", "/docs/transformers/v4.53.2/en/model_doc/gemma2#transformers.Gemma2Config"),
            ("phi", "PhiConfig", ModelCategory.TEXT, [TaskType.TEXT_GENERATION], "Phi model", "/docs/transformers/v4.53.2/en/model_doc/phi#transformers.PhiConfig"),
            ("phi3", "Phi3Config", ModelCategory.TEXT, [TaskType.TEXT_GENERATION], "Phi3 model", "/docs/transformers/v4.53.2/en/model_doc/phi3#transformers.Phi3Config"),
            ("qwen2", "Qwen2Config", ModelCategory.TEXT, [TaskType.TEXT_GENERATION], "Qwen2 model", "/docs/transformers/v4.53.2/en/model_doc/qwen2#transformers.Qwen2Config"),
            ("qwen3", "Qwen3Config", ModelCategory.TEXT, [TaskType.TEXT_GENERATION], "Qwen3 model", "/docs/transformers/v4.53.2/en/model_doc/qwen3#transformers.Qwen3Config"),
            ("mamba", "MambaConfig", ModelCategory.TEXT, [TaskType.TEXT_GENERATION], "Mamba model", "/docs/transformers/v4.53.2/en/model_doc/mamba#transformers.MambaConfig"),
            ("mamba2", "Mamba2Config", ModelCategory.TEXT, [TaskType.TEXT_GENERATION], "mamba2 model", "/docs/transformers/v4.53.2/en/model_doc/mamba2#transformers.Mamba2Config"),
            
            # Seq2Seq 모델들
            ("t5", "T5Config", ModelCategory.TEXT, [TaskType.TEXT2TEXT_GENERATION, TaskType.TRANSLATION, TaskType.SUMMARIZATION], "T5 model", "/docs/transformers/v4.53.2/en/model_doc/t5#transformers.T5Config"),
            ("bart", "BartConfig", ModelCategory.TEXT, [TaskType.TEXT2TEXT_GENERATION, TaskType.SUMMARIZATION], "BART model", "/docs/transformers/v4.53.2/en/model_doc/bart#transformers.BartConfig"),
            ("pegasus", "PegasusConfig", ModelCategory.TEXT, [TaskType.SUMMARIZATION], "Pegasus model", "/docs/transformers/v4.53.2/en/model_doc/pegasus#transformers.PegasusConfig"),
            ("marian", "MarianConfig", ModelCategory.TEXT, [TaskType.TRANSLATION], "Marian model", "/docs/transformers/v4.53.2/en/model_doc/marian#transformers.MarianConfig"),
            ("m2m_100", "M2M100Config", ModelCategory.TEXT, [TaskType.TRANSLATION], "M2M100 model", "/docs/transformers/v4.53.2/en/model_doc/m2m_100#transformers.M2M100Config"),
            ("mbart", "MBartConfig", ModelCategory.TEXT, [TaskType.TEXT2TEXT_GENERATION, TaskType.TRANSLATION], "mBART model", "/docs/transformers/v4.53.2/en/model_doc/mbart#transformers.MBartConfig"),
            ("mt5", "MT5Config", ModelCategory.TEXT, [TaskType.TEXT2TEXT_GENERATION, TaskType.TRANSLATION], "MT5 model", "/docs/transformers/v4.53.2/en/model_doc/mt5#transformers.MT5Config"),
            ("longt5", "LongT5Config", ModelCategory.TEXT, [TaskType.TEXT2TEXT_GENERATION], "LongT5 model", "/docs/transformers/v4.53.2/en/model_doc/longt5#transformers.LongT5Config"),
            ("umt5", "UMT5Config", ModelCategory.TEXT, [TaskType.TEXT2TEXT_GENERATION], "UMT5 model", "/docs/transformers/v4.53.2/en/model_doc/umt5#transformers.UMT5Config"),
            
            # 비전 모델들
            ("vit", "ViTConfig", ModelCategory.VISION, [TaskType.IMAGE_CLASSIFICATION], "ViT model", "/docs/transformers/v4.53.2/en/model_doc/vit#transformers.ViTConfig"),
            ("deit", "DeiTConfig", ModelCategory.VISION, [TaskType.IMAGE_CLASSIFICATION], "DeiT model", "/docs/transformers/v4.53.2/en/model_doc/deit#transformers.DeiTConfig"),
            ("beit", "BeitConfig", ModelCategory.VISION, [TaskType.IMAGE_CLASSIFICATION], "BEiT model", "/docs/transformers/v4.53.2/en/model_doc/beit#transformers.BeitConfig"),
            ("swin", "SwinConfig", ModelCategory.VISION, [TaskType.IMAGE_CLASSIFICATION], "Swin Transformer model", "/docs/transformers/v4.53.2/en/model_doc/swin#transformers.SwinConfig"),
            ("swinv2", "Swinv2Config", ModelCategory.VISION, [TaskType.IMAGE_CLASSIFICATION], "Swin Transformer V2 model", "/docs/transformers/v4.53.2/en/model_doc/swinv2#transformers.Swinv2Config"),
            ("convnext", "ConvNextConfig", ModelCategory.VISION, [TaskType.IMAGE_CLASSIFICATION], "ConvNeXT model", "/docs/transformers/v4.53.2/en/model_doc/convnext#transformers.ConvNextConfig"),
            ("convnextv2", "ConvNextV2Config", ModelCategory.VISION, [TaskType.IMAGE_CLASSIFICATION], "ConvNeXTV2 model", "/docs/transformers/v4.53.2/en/model_doc/convnextv2#transformers.ConvNextV2Config"),
            ("resnet", "ResNetConfig", ModelCategory.VISION, [TaskType.IMAGE_CLASSIFICATION], "ResNet model", "/docs/transformers/v4.53.2/en/model_doc/resnet#transformers.ResNetConfig"),
            ("efficientnet", "EfficientNetConfig", ModelCategory.VISION, [TaskType.IMAGE_CLASSIFICATION], "EfficientNet model", "/docs/transformers/v4.53.2/en/model_doc/efficientnet#transformers.EfficientNetConfig"),
            ("mobilenet_v1", "MobileNetV1Config", ModelCategory.VISION, [TaskType.IMAGE_CLASSIFICATION], "MobileNetV1 model", "/docs/transformers/v4.53.2/en/model_doc/mobilenet_v1#transformers.MobileNetV1Config"),
            ("mobilenet_v2", "MobileNetV2Config", ModelCategory.VISION, [TaskType.IMAGE_CLASSIFICATION], "MobileNetV2 model", "/docs/transformers/v4.53.2/en/model_doc/mobilenet_v2#transformers.MobileNetV2Config"),
            ("mobilevit", "MobileViTConfig", ModelCategory.VISION, [TaskType.IMAGE_CLASSIFICATION], "MobileViT model", "/docs/transformers/v4.53.2/en/model_doc/mobilevit#transformers.MobileViTConfig"),
            ("detr", "DetrConfig", ModelCategory.VISION, [TaskType.OBJECT_DETECTION], "DETR model", "/docs/transformers/v4.53.2/en/model_doc/detr#transformers.DetrConfig"),
            ("deta", "DetaConfig", ModelCategory.VISION, [TaskType.OBJECT_DETECTION], "DETA model", "/docs/transformers/v4.53.2/en/model_doc/deta#transformers.DetaConfig"),
            ("yolos", "YolosConfig", ModelCategory.VISION, [TaskType.OBJECT_DETECTION], "YOLOS model", "/docs/transformers/v4.53.2/en/model_doc/yolos#transformers.YolosConfig"),
            ("segformer", "SegformerConfig", ModelCategory.VISION, [TaskType.IMAGE_SEGMENTATION], "SegFormer model", "/docs/transformers/v4.53.2/en/model_doc/segformer#transformers.SegformerConfig"),
            ("maskformer", "MaskFormerConfig", ModelCategory.VISION, [TaskType.IMAGE_SEGMENTATION], "MaskFormer model", "/docs/transformers/v4.53.2/en/model_doc/maskformer#transformers.MaskFormerConfig"),
            ("mask2former", "Mask2FormerConfig", ModelCategory.VISION, [TaskType.IMAGE_SEGMENTATION], "Mask2Former model", "/docs/transformers/v4.53.2/en/model_doc/mask2former#transformers.Mask2FormerConfig"),
            
            # 오디오 모델들
            ("wav2vec2", "Wav2Vec2Config", ModelCategory.AUDIO, [TaskType.AUTOMATIC_SPEECH_RECOGNITION, TaskType.AUDIO_CLASSIFICATION], "Wav2Vec2 model", "/docs/transformers/v4.53.2/en/model_doc/wav2vec2#transformers.Wav2Vec2Config"),
            ("whisper", "WhisperConfig", ModelCategory.AUDIO, [TaskType.AUTOMATIC_SPEECH_RECOGNITION], "Whisper model", "/docs/transformers/v4.53.2/en/model_doc/whisper#transformers.WhisperConfig"),
            ("hubert", "HubertConfig", ModelCategory.AUDIO, [TaskType.AUTOMATIC_SPEECH_RECOGNITION, TaskType.AUDIO_CLASSIFICATION], "Hubert model", "/docs/transformers/v4.53.2/en/model_doc/hubert#transformers.HubertConfig"),
            ("wavlm", "WavLMConfig", ModelCategory.AUDIO, [TaskType.AUTOMATIC_SPEECH_RECOGNITION, TaskType.AUDIO_CLASSIFICATION], "WavLM model", "/docs/transformers/v4.53.2/en/model_doc/wavlm#transformers.WavLMConfig"),
            ("unispeech", "UniSpeechConfig", ModelCategory.AUDIO, [TaskType.AUTOMATIC_SPEECH_RECOGNITION], "UniSpeech model", "/docs/transformers/v4.53.2/en/model_doc/unispeech#transformers.UniSpeechConfig"),
            ("unispeech-sat", "UniSpeechSatConfig", ModelCategory.AUDIO, [TaskType.AUTOMATIC_SPEECH_RECOGNITION], "UniSpeechSat model", "/docs/transformers/v4.53.2/en/model_doc/unispeech-sat#transformers.UniSpeechSatConfig"),
            ("speech_to_text", "Speech2TextConfig", ModelCategory.AUDIO, [TaskType.AUTOMATIC_SPEECH_RECOGNITION], "Speech2Text model", "/docs/transformers/v4.53.2/en/model_doc/speech_to_text#transformers.Speech2TextConfig"),
            ("speech_to_text_2", "Speech2Text2Config", ModelCategory.AUDIO, [TaskType.AUTOMATIC_SPEECH_RECOGNITION], "Speech2Text2 model", "/docs/transformers/v4.53.2/en/model_doc/speech_to_text_2#transformers.Speech2Text2Config"),
            ("speecht5", "SpeechT5Config", ModelCategory.AUDIO, [TaskType.AUTOMATIC_SPEECH_RECOGNITION, TaskType.TEXT_TO_SPEECH], "SpeechT5 model", "/docs/transformers/v4.53.2/en/model_doc/speecht5#transformers.SpeechT5Config"),
            ("audio-spectrogram-transformer", "ASTConfig", ModelCategory.AUDIO, [TaskType.AUDIO_CLASSIFICATION], "Audio Spectrogram Transformer model", "/docs/transformers/v4.53.2/en/model_doc/audio-spectrogram-transformer#transformers.ASTConfig"),
            
            # 멀티모달 모델들
            ("clip", "CLIPConfig", ModelCategory.MULTIMODAL, [TaskType.ZERO_SHOT_CLASSIFICATION, TaskType.FEATURE_EXTRACTION], "CLIP model", "/docs/transformers/v4.53.2/en/model_doc/clip#transformers.CLIPConfig"),
            ("blip", "BlipConfig", ModelCategory.MULTIMODAL, [TaskType.IMAGE_CLASSIFICATION, TaskType.TEXT_GENERATION], "BLIP model", "/docs/transformers/v4.53.2/en/model_doc/blip#transformers.BlipConfig"),
            ("blip-2", "Blip2Config", ModelCategory.MULTIMODAL, [TaskType.IMAGE_CLASSIFICATION, TaskType.TEXT_GENERATION], "BLIP-2 model", "/docs/transformers/v4.53.2/en/model_doc/blip-2#transformers.Blip2Config"),
            ("llava", "LlavaConfig", ModelCategory.MULTIMODAL, [TaskType.TEXT_GENERATION], "LLaVa model", "/docs/transformers/v4.53.2/en/model_doc/llava#transformers.LlavaConfig"),
            ("llava_next", "LlavaNextConfig", ModelCategory.MULTIMODAL, [TaskType.TEXT_GENERATION], "LLaVA-NeXT model", "/docs/transformers/v4.53.2/en/model_doc/granitevision#transformers.LlavaNextConfig"),
            ("llava_onevision", "LlavaOnevisionConfig", ModelCategory.MULTIMODAL, [TaskType.TEXT_GENERATION], "LLaVA-Onevision model", "/docs/transformers/v4.53.2/en/model_doc/llava_onevision#transformers.LlavaOnevisionConfig"),
            ("paligemma", "PaliGemmaConfig", ModelCategory.MULTIMODAL, [TaskType.TEXT_GENERATION], "PaliGemma model", "/docs/transformers/v4.53.2/en/model_doc/paligemma#transformers.PaliGemmaConfig"),
            ("idefics", "IdeficsConfig", ModelCategory.MULTIMODAL, [TaskType.TEXT_GENERATION], "IDEFICS model", "/docs/transformers/v4.53.2/en/model_doc/idefics#transformers.IdeficsConfig"),
            ("idefics2", "Idefics2Config", ModelCategory.MULTIMODAL, [TaskType.TEXT_GENERATION], "Idefics2 model", "/docs/transformers/v4.53.2/en/model_doc/idefics2#transformers.Idefics2Config"),
            ("kosmos-2", "Kosmos2Config", ModelCategory.MULTIMODAL, [TaskType.TEXT_GENERATION], "KOSMOS-2 model", "/docs/transformers/v4.53.2/en/model_doc/kosmos-2#transformers.Kosmos2Config"),
            ("fuyu", "FuyuConfig", ModelCategory.MULTIMODAL, [TaskType.TEXT_GENERATION], "Fuyu model", "/docs/transformers/v4.53.2/en/model_doc/fuyu#transformers.FuyuConfig"),
            ("chinese_clip", "ChineseCLIPConfig", ModelCategory.MULTIMODAL, [TaskType.ZERO_SHOT_CLASSIFICATION], "Chinese-CLIP model", "/docs/transformers/v4.53.2/en/model_doc/chinese_clip#transformers.ChineseCLIPConfig"),
            ("bridgetower", "BridgeTowerConfig", ModelCategory.MULTIMODAL, [TaskType.ZERO_SHOT_CLASSIFICATION], "BridgeTower model", "/docs/transformers/v4.53.2/en/model_doc/bridgetower#transformers.BridgeTowerConfig"),
            ("align", "AlignConfig", ModelCategory.MULTIMODAL, [TaskType.ZERO_SHOT_CLASSIFICATION], "ALIGN model", "/docs/transformers/v4.53.2/en/model_doc/align#transformers.AlignConfig"),
            ("owlvit", "OwlViTConfig", ModelCategory.MULTIMODAL, [TaskType.OBJECT_DETECTION], "OWL-ViT model", "/docs/transformers/v4.53.2/en/model_doc/owlvit#transformers.OwlViTConfig"),
            ("owlv2", "Owlv2Config", ModelCategory.MULTIMODAL, [TaskType.OBJECT_DETECTION], "OWLv2 model", "/docs/transformers/v4.53.2/en/model_doc/owlv2#transformers.Owlv2Config"),
            ("clap", "ClapConfig", ModelCategory.MULTIMODAL, [TaskType.ZERO_SHOT_CLASSIFICATION], "CLAP model", "/docs/transformers/v4.53.2/en/model_doc/clap#transformers.ClapConfig"),
            ("flava", "FlavaConfig", ModelCategory.MULTIMODAL, [TaskType.IMAGE_CLASSIFICATION, TaskType.TEXT_CLASSIFICATION], "FLAVA model", "/docs/transformers/v4.53.2/en/model_doc/flava#transformers.FlavaConfig"),
            ("lxmert", "LxmertConfig", ModelCategory.MULTIMODAL, [TaskType.QUESTION_ANSWERING], "LXMERT model", "/docs/transformers/v4.53.2/en/model_doc/lxmert#transformers.LxmertConfig"),
            ("layoutlm", "LayoutLMConfig", ModelCategory.MULTIMODAL, [TaskType.TOKEN_CLASSIFICATION], "LayoutLM model", "/docs/transformers/v4.53.2/en/model_doc/layoutlm#transformers.LayoutLMConfig"),
            ("layoutlmv2", "LayoutLMv2Config", ModelCategory.MULTIMODAL, [TaskType.TOKEN_CLASSIFICATION], "LayoutLMv2 model", "/docs/transformers/v4.53.2/en/model_doc/layoutlmv2#transformers.LayoutLMv2Config"),
            ("layoutlmv3", "LayoutLMv3Config", ModelCategory.MULTIMODAL, [TaskType.TOKEN_CLASSIFICATION], "LayoutLMv3 model", "/docs/transformers/v4.53.2/en/model_doc/layoutlmv3#transformers.LayoutLMv3Config"),
            
            # 특수 모델들
            ("encoder-decoder", "EncoderDecoderConfig", ModelCategory.SPECIAL, [TaskType.TEXT2TEXT_GENERATION], "Encoder decoder model", "/docs/transformers/v4.53.2/en/model_doc/encoder-decoder#transformers.EncoderDecoderConfig"),
            ("vision-encoder-decoder", "VisionEncoderDecoderConfig", ModelCategory.SPECIAL, [TaskType.IMAGE_CLASSIFICATION], "Vision Encoder decoder model", "/docs/transformers/v4.53.2/en/model_doc/vision-encoder-decoder#transformers.VisionEncoderDecoderConfig"),
            ("speech-encoder-decoder", "SpeechEncoderDecoderConfig", ModelCategory.SPECIAL, [TaskType.AUTOMATIC_SPEECH_RECOGNITION], "Speech Encoder decoder model", "/docs/transformers/v4.53.2/en/model_doc/speech-encoder-decoder#transformers.SpeechEncoderDecoderConfig"),
            ("vision-text-dual-encoder", "VisionTextDualEncoderConfig", ModelCategory.SPECIAL, [TaskType.FEATURE_EXTRACTION], "VisionTextDualEncoder model", "/docs/transformers/v4.53.2/en/model_doc/vision-text-dual-encoder#transformers.VisionTextDualEncoderConfig"),
            ("decision_transformer", "DecisionTransformerConfig", ModelCategory.SPECIAL, [TaskType.FEATURE_EXTRACTION], "Decision Transformer model", "/docs/transformers/v4.53.2/en/model_doc/decision_transformer#transformers.DecisionTransformerConfig"),
            ("trajectory_transformer", "TrajectoryTransformerConfig", ModelCategory.SPECIAL, [TaskType.FEATURE_EXTRACTION], "Trajectory Transformer model", "/docs/transformers/v4.53.2/en/model_doc/trajectory_transformer#transformers.TrajectoryTransformerConfig"),
            
            # 시계열 모델들
            ("time_series_transformer", "TimeSeriesTransformerConfig", ModelCategory.TIME_SERIES, [TaskType.FEATURE_EXTRACTION], "Time Series Transformer model", "/docs/transformers/v4.53.2/en/model_doc/time_series_transformer#transformers.TimeSeriesTransformerConfig"),
            ("autoformer", "AutoformerConfig", ModelCategory.TIME_SERIES, [TaskType.FEATURE_EXTRACTION], "Autoformer model", "/docs/transformers/v4.53.2/en/model_doc/autoformer#transformers.AutoformerConfig"),
            ("informer", "InformerConfig", ModelCategory.TIME_SERIES, [TaskType.FEATURE_EXTRACTION], "Informer model", "/docs/transformers/v4.53.2/en/model_doc/informer#transformers.InformerConfig"),
            ("patchtst", "PatchTSTConfig", ModelCategory.TIME_SERIES, [TaskType.FEATURE_EXTRACTION], "PatchTST model", "/docs/transformers/v4.53.2/en/model_doc/patchtst#transformers.PatchTSTConfig"),
            ("patchtsmixer", "PatchTSMixerConfig", ModelCategory.TIME_SERIES, [TaskType.FEATURE_EXTRACTION], "PatchTSMixer model", "/docs/transformers/v4.53.2/en/model_doc/patchtsmixer#transformers.PatchTSMixerConfig"),
            ("timesfm", "TimesFmConfig", ModelCategory.TIME_SERIES, [TaskType.FEATURE_EXTRACTION], "TimesFm model", "/docs/transformers/v4.53.2/en/model_doc/timesfm#transformers.TimesFmConfig"),
            
            # 그래프 모델들
            ("graphormer", "GraphormerConfig", ModelCategory.GRAPH, [TaskType.FEATURE_EXTRACTION], "Graphormer model", "/docs/transformers/v4.53.2/en/model_doc/graphormer#transformers.GraphormerConfig"),
        ]
        
        for model_type, config_class, category, tasks, description, doc_link in model_data:
            models[model_type] = ModelTypeInfo(
                model_type=model_type,
                config_class=config_class,
                category=category,
                primary_tasks=tasks,
                description=description,
                doc_link=doc_link
            )
        
        return models
    
    def get_model_info(self, model_type: str) -> Optional[ModelTypeInfo]:
        """모델 타입 정보 조회"""
        return self.models.get(model_type)
    
    def get_models_by_category(self, category: ModelCategory) -> List[ModelTypeInfo]:
        """카테고리별 모델 목록 조회"""
        return [info for info in self.models.values() if info.category == category]
    
    def get_models_by_task(self, task: TaskType) -> List[ModelTypeInfo]:
        """태스크별 모델 목록 조회"""
        return [info for info in self.models.values() if task in info.primary_tasks]
    
    def search_models(self, query: str) -> List[ModelTypeInfo]:
        """모델 검색"""
        query = query.lower()
        results = []
        
        for info in self.models.values():
            if (query in info.model_type.lower() or 
                query in info.description.lower() or
                query in info.config_class.lower()):
                results.append(info)
        
        return results
    
    def get_all_model_types(self) -> List[str]:
        """모든 모델 타입 목록 반환"""
        return list(self.models.keys())
    
    def get_task_to_model_class_mapping(self) -> Dict[str, str]:
        """태스크별 Auto 모델 클래스 매핑"""
        task_to_class = {
            TaskType.TEXT_CLASSIFICATION.value: "AutoModelForSequenceClassification",
            TaskType.TOKEN_CLASSIFICATION.value: "AutoModelForTokenClassification",
            TaskType.QUESTION_ANSWERING.value: "AutoModelForQuestionAnswering",
            TaskType.TEXT_GENERATION.value: "AutoModelForCausalLM",
            TaskType.TEXT2TEXT_GENERATION.value: "AutoModelForSeq2SeqLM",
            TaskType.TRANSLATION.value: "AutoModelForSeq2SeqLM",
            TaskType.SUMMARIZATION.value: "AutoModelForSeq2SeqLM",
            TaskType.FILL_MASK.value: "AutoModelForMaskedLM",
            TaskType.FEATURE_EXTRACTION.value: "AutoModel",
            TaskType.SENTENCE_SIMILARITY.value: "AutoModel",
            TaskType.IMAGE_CLASSIFICATION.value: "AutoModelForImageClassification",
            TaskType.OBJECT_DETECTION.value: "AutoModelForObjectDetection",
            TaskType.IMAGE_SEGMENTATION.value: "AutoModelForImageSegmentation",
            TaskType.AUTOMATIC_SPEECH_RECOGNITION.value: "AutoModelForSpeechSeq2Seq",
            TaskType.AUDIO_CLASSIFICATION.value: "AutoModelForAudioClassification",
            TaskType.TEXT_TO_SPEECH.value: "AutoModelForTextToWaveform",
            TaskType.ZERO_SHOT_CLASSIFICATION.value: "AutoModelForZeroShotImageClassification",
        }
        return task_to_class
    
    def is_supported_model(self, model_type: str) -> bool:
        """지원되는 모델인지 확인"""
        return model_type in self.models
    
    def get_model_categories(self) -> List[ModelCategory]:
        """모든 모델 카테고리 반환"""
        return list(ModelCategory)
    
    def get_task_types(self) -> List[TaskType]:
        """모든 태스크 타입 반환"""
        return list(TaskType)

# 전역 인스턴스
model_database = HuggingFaceModelDatabase()