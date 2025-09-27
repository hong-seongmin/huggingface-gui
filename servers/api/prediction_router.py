"""
Prediction API router for FastAPI server.

This module contains router endpoints for model prediction operations,
extracted from the original fastapi_server.py file.
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
from datetime import datetime
import logging
import torch
import re

from .models import PredictionRequest, PredictionResponse


class PredictionRouter:
    """Router class for model prediction endpoints."""

    def __init__(self, model_manager, device_manager=None, logger=None):
        """Initialize prediction router."""
        self.model_manager = model_manager
        self.device_manager = device_manager
        self.logger = logger or logging.getLogger("PredictionRouter")
        self.router = APIRouter(prefix="/models", tags=["prediction"])
        self._setup_routes()

    def _setup_routes(self):
        """Setup prediction routes."""

        @self.router.post("/{model_name}/predict")
        async def predict(model_name: str, request: PredictionRequest):
            """Make prediction using specified model."""
            try:
                start_time = datetime.now()

                # Text preprocessing
                processed_text = self._preprocess_text(request.text)
                if processed_text != request.text:
                    self.logger.info(f"Text preprocessing applied: '{request.text}' -> '{processed_text}'")
                    request.text = processed_text

                # Check model status
                model_info = self.model_manager.get_model_info(model_name)
                if not model_info or model_info.status != "loaded":
                    raise HTTPException(status_code=404, detail="Model not found or not loaded")

                # Get model and tokenizer
                model_tokenizer = self.model_manager.get_model_for_inference(model_name)
                if not model_tokenizer:
                    raise HTTPException(status_code=500, detail="Failed to get model for inference")

                model, tokenizer = model_tokenizer

                # Get supported tasks
                available_tasks = self.model_manager.get_available_tasks(model_name)
                if not available_tasks:
                    raise HTTPException(status_code=400, detail="No supported tasks found for this model")

                # Determine task
                if "sentiment" in model_name.lower() or "classifier" in model_name.lower():
                    task = "text-classification"
                else:
                    task = available_tasks[0]

                # Perform inference
                try:
                    result = self._unified_inference(model, tokenizer, request.text, task, request)
                    processing_time = (datetime.now() - start_time).total_seconds()

                    return PredictionResponse(
                        success=True,
                        result=result,
                        model_name=model_name,
                        processing_time=processing_time,
                        metadata={
                            "task": task,
                            "input_length": len(request.text),
                            "timestamp": datetime.now().isoformat()
                        }
                    )

                except Exception as e:
                    # Fallback mechanism
                    self.logger.warning(f"Primary prediction failed: {e}")
                    try:
                        fallback_result = self._fallback_prediction(model_name, request.text, task)
                        processing_time = (datetime.now() - start_time).total_seconds()

                        return PredictionResponse(
                            success=True,
                            result=fallback_result,
                            model_name=model_name,
                            processing_time=processing_time,
                            metadata={
                                "task": task,
                                "fallback_used": True,
                                "timestamp": datetime.now().isoformat()
                            }
                        )
                    except Exception as fallback_error:
                        self.logger.error(f"Fallback prediction also failed: {fallback_error}")
                        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error in prediction endpoint: {e}")
                raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

    def _preprocess_text(self, text: str) -> str:
        """Preprocess input text."""
        if not text or not isinstance(text, str):
            return text

        try:
            # 1. Handle escape sequences
            processed = text.replace('\\n', '\n').replace('\\r', '\r').replace('\\t', '\t')

            # 2. Normalize whitespace
            processed = re.sub(r'\s+', ' ', processed)

            # 3. Strip whitespace
            processed = processed.strip()

            # 4. Normalize special characters
            processed = processed.replace(''', "'").replace(''', "'")
            processed = processed.replace('"', '"').replace('"', '"')

            # 5. Check for empty string
            if not processed:
                return text

            return processed

        except Exception as e:
            self.logger.warning(f"Text preprocessing failed: {e}")
            return text

    def _unified_inference(self, model: Any, tokenizer: Any, text: str, task: str, request: PredictionRequest) -> Any:
        """Unified inference engine for all tasks."""
        try:
            # Ensure device consistency if device_manager is available
            if self.device_manager:
                model, tokenizer = self.device_manager.ensure_device_consistency(model, tokenizer)

            # Tokenize input
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

            # Move inputs to model device if device_manager is available
            if self.device_manager:
                inputs = self.device_manager.prepare_inputs(inputs, model)

            # Validate device consistency if device_manager is available
            if self.device_manager and not self.device_manager.validate_device_consistency(model):
                raise Exception("Model device consistency validation failed")

            with torch.no_grad():
                outputs = model(**inputs)

                if task == "text-classification":
                    return self._process_classification_output(outputs, model)
                elif task == "feature-extraction":
                    return self._process_embedding_output(outputs, inputs)
                elif task == "text-generation":
                    return self._process_generation_output(outputs, tokenizer, request)
                else:
                    # Default processing
                    if hasattr(outputs, 'last_hidden_state'):
                        return outputs.last_hidden_state.cpu().numpy().tolist()
                    elif hasattr(outputs, 'logits'):
                        return outputs.logits.cpu().numpy().tolist()
                    else:
                        return str(outputs)

        except Exception as e:
            self.logger.error(f"Unified inference failed: {e}")
            raise Exception(f"Unified inference failed: {str(e)}")

    def _process_classification_output(self, outputs: Any, model: Any) -> List[Dict[str, Any]]:
        """Process text classification output."""
        if hasattr(outputs, 'logits'):
            import torch.nn.functional as F
            probs = F.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(outputs.logits, dim=-1)

            # Get label mapping
            if hasattr(model.config, 'id2label'):
                id2label = model.config.id2label
                predicted_label = id2label.get(predicted_class.item(), f"LABEL_{predicted_class.item()}")
            else:
                predicted_label = f"LABEL_{predicted_class.item()}"

            return [{
                "label": predicted_label,
                "score": probs[0][predicted_class.item()].item()
            }]
        else:
            raise Exception(f"Classification model output does not contain logits: {outputs}")

    def _process_embedding_output(self, outputs: Any, inputs: Dict[str, torch.Tensor]) -> List[List[float]]:
        """Process embedding output."""
        if hasattr(outputs, 'last_hidden_state'):
            # Use average pooling
            attention_mask = inputs.get('attention_mask')
            if attention_mask is not None:
                # Average excluding masked tokens
                masked_embeddings = outputs.last_hidden_state * attention_mask.unsqueeze(-1)
                pooled_embeddings = masked_embeddings.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
            else:
                # Simple average
                pooled_embeddings = outputs.last_hidden_state.mean(dim=1)

            return pooled_embeddings.cpu().numpy().tolist()
        elif hasattr(outputs, 'pooler_output'):
            return outputs.pooler_output.cpu().numpy().tolist()
        else:
            raise Exception(f"Embedding model output does not contain expected fields: {outputs}")

    def _process_generation_output(self, outputs: Any, tokenizer: Any, request: PredictionRequest) -> List[Dict[str, str]]:
        """Process text generation output."""
        if hasattr(outputs, 'logits'):
            # Simple next token prediction
            next_token_id = torch.argmax(outputs.logits[0, -1, :]).item()
            next_token = tokenizer.decode([next_token_id])
            return [{"generated_text": next_token}]
        else:
            raise Exception(f"Generation model output does not contain logits: {outputs}")

    def _fallback_prediction(self, model_name: str, text: str, task: str) -> Dict[str, Any]:
        """Fallback prediction mechanism."""
        try:
            if task == "text-classification":
                return [
                    {"label": "UNKNOWN", "score": 0.5},
                    {"label": "NEUTRAL", "score": 0.5}
                ]
            elif task == "token-classification":
                tokens = text.split()
                return [
                    {"entity": "O", "score": 0.5, "index": i, "word": token, "start": 0, "end": len(token)}
                    for i, token in enumerate(tokens)
                ]
            elif task == "text-generation":
                return [{"generated_text": f"{text} [Generation failed - using fallback]"}]
            elif task == "summarization":
                return [{"summary_text": f"Summary of: {text[:50]}..."}]
            else:
                return {"result": f"Fallback response for task: {task}", "input": text}

        except Exception as e:
            self.logger.error(f"Fallback mechanism failed: {e}")
            return {"error": "Complete prediction failure", "input": text}

    def get_router(self) -> APIRouter:
        """Get the configured router instance."""
        return self.router