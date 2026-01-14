"""Multimodal Encoder Module
Unified processing for text, image, and audio inputs using GPT-4o and specialized models
"""

from typing import Dict, Any, Optional, Union, List
import logging
from dataclasses import dataclass
from datetime import datetime
import base64
import os
from io import BytesIO
from PIL import Image

import torch
from transformers import CLIPProcessor, CLIPModel, WhisperProcessor, WhisperForConditionalGeneration

# 配置日志
logger = logging.getLogger(__name__)


def get_openai_client():
    """获取OpenAI客户端（延迟导入以避免初始化问题）"""
    from openai import OpenAI
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key and openai_api_key != "<your-openai-api-key>":
        try:
            return OpenAI(api_key=openai_api_key)
        except Exception as e:
            logger.error(f"OpenAI客户端初始化失败: {str(e)}")
    return None


@dataclass
class MultimodalInput:
    """多模态输入数据结构"""
    text: Optional[str] = None
    images: Optional[List[Union[str, BytesIO]]] = None  # 可以是文件路径或BytesIO对象
    audio: Optional[Union[str, BytesIO]] = None  # 可以是文件路径或BytesIO对象
    timestamp: datetime = datetime.now()
    context: Optional[Dict[str, Any]] = None


@dataclass
class EncodedFeature:
    """编码后的特征数据结构"""
    modality: str  # text, image, audio
    encoding: Any  # 具体的编码数据，可能是嵌入向量或文本描述
    confidence: float
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class UnifiedEncoding:
    """统一编码结果"""
    confidence: float
    timestamp: datetime
    text_encoding: Optional[EncodedFeature] = None
    image_encodings: Optional[List[EncodedFeature]] = None
    audio_encoding: Optional[EncodedFeature] = None
    unified_representation: Optional[Any] = None  # 跨模态统一表示


class MultimodalEncoder:
    """多模态编码器，统一处理文本、图像和音频输入"""
    
    def __init__(self):
        self.openai_client = get_openai_client()
        self.use_openai = self.openai_client is not None
        
        # 初始化本地模型
        self.clip_model = None
        self.clip_processor = None
        self.whisper_model = None
        self.whisper_processor = None
        
        # 初始化模型
        self._init_local_models()
        
        logger.info(f"多模态编码器初始化完成，OpenAI支持: {self.use_openai}")
    
    def _init_local_models(self):
        """初始化本地模型"""
        try:
            # 初始化CLIP模型用于图像编码
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            logger.info("✅ CLIP模型初始化成功")
        except Exception as e:
            logger.warning(f"⚠️ CLIP模型初始化失败: {str(e)}")
        
        try:
            # 初始化Whisper模型用于音频编码
            self.whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
            self.whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-small")
            logger.info("✅ Whisper模型初始化成功")
        except Exception as e:
            logger.warning(f"⚠️ Whisper模型初始化失败: {str(e)}")
    
    def encode_text(self, text: str) -> EncodedFeature:
        """编码文本输入"""
        timestamp = datetime.now()
        
        if self.use_openai:
            # 使用OpenAI的嵌入API
            try:
                response = self.openai_client.embeddings.create(
                    input=text,
                    model="text-embedding-3-small"
                )
                embedding = response.data[0].embedding
                return EncodedFeature(
                    modality="text",
                    encoding=embedding,
                    confidence=1.0,
                    timestamp=timestamp,
                    metadata={"model": "text-embedding-3-small"}
                )
            except Exception as e:
                logger.error(f"OpenAI文本编码失败: {str(e)}")
        
        # 回退到简单的文本处理
        return EncodedFeature(
            modality="text",
            encoding=text,
            confidence=0.8,
            timestamp=timestamp,
            metadata={"model": "local"}
        )
    
    def _load_image(self, image_input: Union[str, BytesIO]) -> Image.Image:
        """加载图像"""
        if isinstance(image_input, str):
            # 从文件路径加载
            if os.path.exists(image_input):
                return Image.open(image_input).convert("RGB")
            else:
                raise FileNotFoundError(f"图像文件不存在: {image_input}")
        elif isinstance(image_input, BytesIO):
            # 从BytesIO加载
            image_input.seek(0)
            return Image.open(image_input).convert("RGB")
        else:
            raise TypeError(f"不支持的图像类型: {type(image_input)}")
    
    def encode_image(self, image_input: Union[str, BytesIO]) -> EncodedFeature:
        """编码图像输入"""
        timestamp = datetime.now()
        
        try:
            image = self._load_image(image_input)
            
            if self.clip_model and self.clip_processor:
                # 使用CLIP模型编码图像
                inputs = self.clip_processor(images=image, return_tensors="pt")
                with torch.no_grad():
                    image_features = self.clip_model.get_image_features(**inputs)
                
                return EncodedFeature(
                    modality="image",
                    encoding=image_features.squeeze().tolist(),
                    confidence=0.9,
                    timestamp=timestamp,
                    metadata={"model": "clip-vit-base-patch32"}
                )
            else:
                # 回退到图像描述生成（如果有OpenAI）
                if self.use_openai:
                    try:
                        # 将图像转换为base64
                        buffered = BytesIO()
                        image.save(buffered, format="PNG")
                        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                        
                        # 调用GPT-4o生成图像描述
                        response = self.openai_client.chat.completions.create(
                            model="gpt-4o",
                            messages=[
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": "请描述这张图像的内容，特别是与农业相关的元素"},
                                        {
                                            "type": "image_url",
                                            "image_url": {
                                                "url": f"data:image/png;base64,{img_str}"
                                            }
                                        }
                                    ]
                                }
                            ],
                            temperature=0.1,
                            max_tokens=500
                        )
                        
                        description = response.choices[0].message.content
                        return EncodedFeature(
                            modality="image",
                            encoding=description,
                            confidence=0.95,
                            timestamp=timestamp,
                            metadata={"model": "gpt-4o", "method": "description"}
                        )
                    except Exception as e:
                        logger.error(f"OpenAI图像编码失败: {str(e)}")
        except Exception as e:
            logger.error(f"图像编码失败: {str(e)}")
        
        # 最终回退
        return EncodedFeature(
            modality="image",
            encoding="无法处理的图像",
            confidence=0.1,
            timestamp=timestamp,
            metadata={"model": "fallback"}
        )
    
    def encode_audio(self, audio_input: Union[str, BytesIO]) -> EncodedFeature:
        """编码音频输入"""
        timestamp = datetime.now()
        
        if self.whisper_model and self.whisper_processor:
            # 使用Whisper模型进行语音转文字
            try:
                if isinstance(audio_input, str):
                    # 从文件加载
                    inputs = self.whisper_processor(audio=audio_input, return_tensors="pt")
                else:
                    # 从BytesIO加载
                    audio_input.seek(0)
                    inputs = self.whisper_processor(audio=audio_input.read(), return_tensors="pt")
                
                with torch.no_grad():
                    predicted_ids = self.whisper_model.generate(**inputs)
                
                transcription = self.whisper_processor.batch_decode(
                    predicted_ids, skip_special_tokens=True
                )[0]
                
                return EncodedFeature(
                    modality="audio",
                    encoding=transcription,
                    confidence=0.9,
                    timestamp=timestamp,
                    metadata={"model": "whisper-small", "method": "transcription"}
                )
            except Exception as e:
                logger.error(f"Whisper音频编码失败: {str(e)}")
        
        # 回退到OpenAI的音频转文字（如果可用）
        if self.use_openai and isinstance(audio_input, str) and os.path.exists(audio_input):
            try:
                with open(audio_input, "rb") as audio_file:
                    response = self.openai_client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        language="zh"
                    )
                
                return EncodedFeature(
                    modality="audio",
                    encoding=response.text,
                    confidence=0.95,
                    timestamp=timestamp,
                    metadata={"model": "whisper-1", "method": "transcription"}
                )
            except Exception as e:
                logger.error(f"OpenAI音频编码失败: {str(e)}")
        
        # 最终回退
        return EncodedFeature(
            modality="audio",
            encoding="无法处理的音频",
            confidence=0.1,
            timestamp=timestamp,
            metadata={"model": "fallback"}
        )
    
    def encode_multimodal(self, input_data: MultimodalInput) -> UnifiedEncoding:
        """编码多模态输入，生成统一表示"""
        logger.debug(f"开始编码多模态输入: 文本={input_data.text is not None}, "
                   f"图像={len(input_data.images) if input_data.images else 0}, "
                   f"音频={input_data.audio is not None}")
        
        text_encoding = None
        image_encodings = []
        audio_encoding = None
        
        # 编码文本
        if input_data.text:
            text_encoding = self.encode_text(input_data.text)
        
        # 编码图像
        if input_data.images:
            for image in input_data.images:
                img_encoding = self.encode_image(image)
                image_encodings.append(img_encoding)
        
        # 编码音频
        if input_data.audio:
            audio_encoding = self.encode_audio(input_data.audio)
        
        # 生成统一表示
        unified_representation = self._generate_unified_representation(
            text_encoding, image_encodings, audio_encoding
        )
        
        # 计算整体置信度
        confidences = []
        if text_encoding:
            confidences.append(text_encoding.confidence)
        for img_enc in image_encodings:
            confidences.append(img_enc.confidence)
        if audio_encoding:
            confidences.append(audio_encoding.confidence)
        
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return UnifiedEncoding(
            text_encoding=text_encoding,
            image_encodings=image_encodings,
            audio_encoding=audio_encoding,
            unified_representation=unified_representation,
            confidence=avg_confidence,
            timestamp=input_data.timestamp
        )
    
    def _generate_unified_representation(self, 
                                        text_encoding: Optional[EncodedFeature],
                                        image_encodings: List[EncodedFeature],
                                        audio_encoding: Optional[EncodedFeature]) -> Any:
        """生成跨模态统一表示"""
        # 简单的统一表示方法：将所有编码转换为文本描述，然后组合
        descriptions = []
        
        if text_encoding:
            if isinstance(text_encoding.encoding, str):
                descriptions.append(f"文本: {text_encoding.encoding}")
            else:
                # 如果是嵌入向量，转换为简单描述
                descriptions.append(f"文本内容（已编码）")
        
        for i, img_enc in enumerate(image_encodings):
            if isinstance(img_enc.encoding, str):
                descriptions.append(f"图像{i+1}: {img_enc.encoding}")
            else:
                descriptions.append(f"图像{i+1}（已编码）")
        
        if audio_encoding:
            if isinstance(audio_encoding.encoding, str):
                descriptions.append(f"音频: {audio_encoding.encoding}")
            else:
                descriptions.append(f"音频（已编码）")
        
        # 组合所有描述
        return " ".join(descriptions)
    
    def process_with_gpt4o(self, encoding: UnifiedEncoding, prompt: str) -> str:
        """使用GPT-4o处理统一编码结果"""
        if not self.openai_client:
            logger.error("GPT-4o不可用，无法处理多模态输入")
            return "无法处理多模态输入，GPT-4o不可用"
        
        try:
            messages = [
                {
                    "role": "system",
                    "content": "你是一个智能农业AI助手，能够理解多模态输入并提供专业的农业决策建议。"
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # 添加统一表示作为上下文
            if encoding.unified_representation:
                messages[1]["content"].append(
                    {"type": "text", "text": f"上下文信息: {encoding.unified_representation}"}
                )
            
            # 调用GPT-4o API
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.7,
                max_tokens=2000
            )
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"GPT-4o处理失败: {str(e)}")
            return f"处理多模态输入时出错: {str(e)}"


# 创建全局多模态编码器实例
multimodal_encoder = MultimodalEncoder()