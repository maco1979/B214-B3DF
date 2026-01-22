"""
AI推理引擎
高性能推理服务，支持批量处理、边缘计算和实时推理
"""

import time
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

# 条件导入 JAX 和 Flax（处理 Python 3.14 兼容性问题）
_jax_available = False
_flax_available = False

try:
    import jax
    import jax.numpy as jnp
    _jax_available = True
except ImportError as e:
    logging.warning(f"JAX 不可用: {e}")
    jax = None
    jnp = None

try:
    from flax.training import train_state
    _flax_available = True
except (ImportError, TypeError) as e:
    logging.warning(f"Flax 不可用: {e}")
    train_state = None

# 条件导入模型（可能依赖 JAX/Flax）
try:
    from ..models import TransformerModel, VisionModel, DiffusionModel
except (ImportError, TypeError) as e:
    logging.warning(f"模型导入失败: {e}")
    TransformerModel = None
    VisionModel = None
    DiffusionModel = None

from .model_manager import ModelManager


class InferenceResult:
    """推理结果"""
    
    def __init__(
        self,
        predictions: Any,
        confidence: Optional[float] = None,
        processing_time: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.predictions = predictions
        self.confidence = confidence
        self.processing_time = processing_time
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {
            "predictions": self.predictions.tolist() if hasattr(self.predictions, 'tolist') else self.predictions,
            "processing_time": self.processing_time,
            "metadata": self.metadata
        }
        
        if self.confidence is not None:
            result["confidence"] = self.confidence
        
        return result


class InferenceEngine:
    """AI推理引擎"""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        
        # 缓存已编译的推理函数
        self._compiled_functions: Dict[str, Any] = {}
        
        # 推理统计
        self._inference_stats: Dict[str, Dict[str, Any]] = {}
    
    async def text_generation(
        self,
        model_id: str,
        prompt: str,
        max_length: int = 100,
        temperature: float = 1.0,
        repetition_penalty: float = 1.0,
        num_return_sequences: int = 1,
        beam_search: bool = False,
        beam_width: int = 5,
        early_stopping: bool = True,
        no_repeat_ngram_size: int = 0,
        do_sample: bool = False,
        top_p: float = 1.0,
        top_k: int = 0
    ) -> InferenceResult:
        """文本生成推理"""
        
        start_time = time.time()
        
        # 加载模型
        load_result = await self.model_manager.load_model(model_id)
        if not load_result["success"]:
            raise ValueError(load_result["error"])
        state = load_result["model"]
        metadata = load_result["metadata"]
        
        if metadata["type"] != "transformer":
            raise ValueError("文本生成需要Transformer模型")
        
        # 编译推理函数（如果尚未编译）
        if model_id not in self._compiled_functions:
            self._compile_text_generation_function(state, model_id)
        
        # 预处理输入
        tokenized_prompt = self._tokenize_text(prompt, metadata)
        
        # 执行推理
        rng = jax.random.PRNGKey(int(time.time()))
        generated_tokens = self._compiled_functions[model_id](
            state.params, rng, tokenized_prompt, max_length, temperature, repetition_penalty,
            beam_search, beam_width, early_stopping, no_repeat_ngram_size, do_sample, top_p, top_k
        )
        
        # 后处理输出
        generated_text = self._detokenize_text(generated_tokens, metadata)
        
        processing_time = time.time() - start_time
        
        # 更新统计信息
        self._update_stats(model_id, processing_time)
        
        return InferenceResult(
            predictions=generated_text,
            confidence=None,  # 文本生成没有置信度
            processing_time=processing_time,
            metadata={
                "model_id": model_id,
                "input_length": len(tokenized_prompt),
                "output_length": len(generated_tokens)
            }
        )
    
    def _compile_text_generation_function(self, state: train_state.TrainState, model_id: str):
        """编译文本生成函数"""
        
        def generate_fn(params, rng, prompt, max_length, temperature, repetition_penalty=1.0, beam_search=False, beam_width=5, early_stopping=True, no_repeat_ngram_size=0, do_sample=False, top_p=1.0, top_k=0):
            # 创建模型实例
            model = TransformerModel(**state.params["params"].keys())  # 简化实现
            
            # 生成文本
            generated = model.generate(params, rng, prompt, max_length, temperature, repetition_penalty, beam_search, beam_width, early_stopping, no_repeat_ngram_size, do_sample, top_p, top_k)
            return generated
        
        # 编译函数
        self._compiled_functions[model_id] = jax.jit(generate_fn)
    
    async def image_classification(
        self,
        model_id: str,
        image: jnp.ndarray,
        top_k: int = 5
    ) -> InferenceResult:
        """图像分类推理"""
        
        start_time = time.time()
        
        # 加载模型
        load_result = await self.model_manager.load_model(model_id)
        if not load_result["success"]:
            raise ValueError(load_result["error"])
        state = load_result["model"]
        metadata = load_result["metadata"]
        
        if metadata["type"] != "vision":
            raise ValueError("图像分类需要视觉模型")
        
        # 编译推理函数
        if model_id not in self._compiled_functions:
            self._compile_image_classification_function(state, model_id)
        
        # 预处理图像
        processed_image = self._preprocess_image(image, metadata)
        
        # 执行推理
        logits = self._compiled_functions[model_id](state.params, processed_image)
        
        # 计算概率和置信度
        probabilities = jax.nn.softmax(logits, axis=-1)
        top_probs, top_indices = jax.lax.top_k(probabilities[0], top_k)
        
        processing_time = time.time() - start_time
        
        # 格式化结果
        predictions = [
            {"class_id": int(idx), "confidence": float(prob), "class_name": f"class_{idx}"}
            for prob, idx in zip(top_probs, top_indices)
        ]
        
        # 更新统计信息
        self._update_stats(model_id, processing_time)
        
        return InferenceResult(
            predictions=predictions,
            confidence=float(top_probs[0]),  # 最高置信度
            processing_time=processing_time,
            metadata={
                "model_id": model_id,
                "input_shape": image.shape,
                "top_k": top_k
            }
        )
    
    def _compile_image_classification_function(self, state: train_state.TrainState, model_id: str):
        """编译图像分类函数"""
        
        def classify_fn(params, image):
            # 创建模型实例
            model = VisionModel(**state.params["params"].keys())  # 简化实现
            
            # 前向传播
            logits = model.apply(params, image)
            return logits
        
        # 编译函数
        self._compiled_functions[model_id] = jax.jit(classify_fn)
    
    async def image_generation(
        self,
        model_id: str,
        num_samples: int = 1,
        image_size: int = 256,
        guidance_scale: float = 7.5
    ) -> InferenceResult:
        """图像生成推理"""
        
        start_time = time.time()
        
        # 加载模型
        load_result = await self.model_manager.load_model(model_id)
        if not load_result["success"]:
            raise ValueError(load_result["error"])
        state = load_result["model"]
        metadata = load_result["metadata"]
        
        if metadata["type"] != "diffusion":
            raise ValueError("图像生成需要扩散模型")
        
        # 编译推理函数
        if model_id not in self._compiled_functions:
            self._compile_image_generation_function(state, model_id)
        
        # 执行推理
        rng = jax.random.PRNGKey(int(time.time()))
        generated_images = self._compiled_functions[model_id](
            state.params, rng, num_samples, image_size
        )
        
        processing_time = time.time() - start_time
        
        # 更新统计信息
        self._update_stats(model_id, processing_time)
        
        return InferenceResult(
            predictions=generated_images,
            confidence=None,  # 图像生成没有置信度
            processing_time=processing_time,
            metadata={
                "model_id": model_id,
                "num_samples": num_samples,
                "image_size": image_size,
                "guidance_scale": guidance_scale
            }
        )
    
    def _compile_image_generation_function(self, state: train_state.TrainState, model_id: str):
        """编译图像生成函数"""
        
        def generate_fn(params, rng, num_samples, image_size):
            # 创建模型实例
            model = DiffusionModel(**state.params["params"].keys())  # 简化实现
            
            # 生成图像
            generated = model.sample(params, rng, num_samples, image_size)
            return generated
        
        # 编译函数
        self._compiled_functions[model_id] = jax.jit(generate_fn)
    
    async def batch_inference(
        self,
        model_id: str,
        inputs: List[Any],
        batch_size: int = 32
    ) -> List[InferenceResult]:
        """批量推理"""
        
        results = []
        
        for i in range(0, len(inputs), batch_size):
            batch_inputs = inputs[i:i+batch_size]
            
            # 根据模型类型执行批量推理
            # 使用load_model获取metadata，因为get_model_info可能不存在
            load_result = await self.model_manager.load_model(model_id)
            if not load_result["success"]:
                raise ValueError(load_result["error"])
            metadata = load_result["metadata"]
            
            if metadata["type"] == "vision":
                batch_results = await self._batch_image_classification(model_id, batch_inputs)
            elif metadata["type"] == "transformer":
                batch_results = await self._batch_text_generation(model_id, batch_inputs)
            else:
                raise ValueError(f"不支持的批量推理模型类型: {metadata['type']}")
            
            results.extend(batch_results)
        
        return results
    
    async def _batch_image_classification(self, model_id: str, images: List[jnp.ndarray]) -> List[InferenceResult]:
        """批量图像分类"""
        
        batch_results = []
        
        for image in images:
            try:
                result = await self.image_classification(model_id, image)
                batch_results.append(result)
            except Exception as e:
                # 创建错误结果
                error_result = InferenceResult(
                    predictions=None,
                    confidence=0.0,
                    processing_time=0.0,
                    metadata={"error": str(e)}
                )
                batch_results.append(error_result)
        
        return batch_results
    
    async def _batch_text_generation(self, model_id: str, texts: List[str]) -> List[InferenceResult]:
        """批量文本生成"""
        
        batch_results = []
        
        for text in texts:
            try:
                result = await self.text_generation(model_id, text)
                batch_results.append(result)
            except Exception as e:
                # 创建错误结果
                error_result = InferenceResult(
                    predictions=None,
                    confidence=0.0,
                    processing_time=0.0,
                    metadata={"error": str(e)}
                )
                batch_results.append(error_result)
        
        return batch_results
    
    def _tokenize_text(self, text: str, metadata: Dict[str, Any]) -> jnp.ndarray:
        """文本分词（简化实现）"""
        # 实际应用中需要使用专业的分词器
        vocab_size = metadata.get("metadata", {}).get("vocab_size", 1000)
        
        # 简单的字符级分词
        tokens = [ord(c) % vocab_size for c in text]
        return jnp.array(tokens, dtype=jnp.int32)
    
    def _detokenize_text(self, tokens: jnp.ndarray, metadata: Dict[str, Any]) -> str:
        """文本去分词（简化实现）"""
        # 简单的字符级去分词
        return ''.join(chr(int(token)) for token in tokens if 0 <= token < 65536)
    
    def _preprocess_image(self, image: jnp.ndarray, metadata: Dict[str, Any]) -> jnp.ndarray:
        """图像预处理"""
        
        # 调整大小
        target_size = metadata.get("metadata", {}).get("image_size", 224)
        if image.shape[1] != target_size or image.shape[2] != target_size:
            image = jax.image.resize(
                image,
                shape=(image.shape[0], target_size, target_size, image.shape[3]),
                method='bilinear'
            )
        
        # 归一化
        image = image.astype(jnp.float32) / 255.0
        
        # 标准化（ImageNet统计）
        mean = jnp.array([0.485, 0.456, 0.406]).reshape(1, 1, 1, 3)
        std = jnp.array([0.229, 0.224, 0.225]).reshape(1, 1, 1, 3)
        image = (image - mean) / std
        
        return image
    
    def _update_stats(self, model_id: str, processing_time: float):
        """更新推理统计信息"""
        
        if model_id not in self._inference_stats:
            self._inference_stats[model_id] = {
                "total_inferences": 0,
                "total_processing_time": 0.0,
                "avg_processing_time": 0.0,
                "last_inference_time": None
            }
        
        stats = self._inference_stats[model_id]
        stats["total_inferences"] += 1
        stats["total_processing_time"] += processing_time
        stats["avg_processing_time"] = stats["total_processing_time"] / stats["total_inferences"]
        stats["last_inference_time"] = time.time()
    
    def get_inference_stats(self, model_id: str) -> Dict[str, Any]:
        """获取推理统计信息"""
        
        if model_id in self._inference_stats:
            return self._inference_stats[model_id].copy()
        else:
            return {
                "total_inferences": 0,
                "total_processing_time": 0.0,
                "avg_processing_time": 0.0,
                "last_inference_time": None
            }
    
    def clear_cache(self, model_id: Optional[str] = None):
        """清除缓存"""
        
        if model_id is None:
            self._compiled_functions.clear()
            self._inference_stats.clear()
        elif model_id in self._compiled_functions:
            del self._compiled_functions[model_id]
        if model_id in self._inference_stats:
            del self._inference_stats[model_id]