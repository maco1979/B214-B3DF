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

# 导入Bi-ATEN模块
from .bi_atten_module import bi_aten_module, DomainGeneralizationResult

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
    text_encoding: Optional[EncodedFeature] = None
    image_encodings: Optional[List[EncodedFeature]] = None
    audio_encoding: Optional[EncodedFeature] = None
    unified_representation: Optional[Any] = None  # 跨模态统一表示
    timestamp: datetime = datetime.now()


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
            confidence=avg_confidence,
            text_encoding=text_encoding,
            image_encodings=image_encodings,
            audio_encoding=audio_encoding,
            unified_representation=unified_representation,
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
    
    def _optimize_prompt(self, prompt: str) -> str:
        """优化提示词，减少冗余信息，降低token消耗"""
        # 移除重复的词语和短语
        words = prompt.split()
        optimized_words = []
        seen = set()
        for word in words:
            if word not in seen:
                seen.add(word)
                optimized_words.append(word)
        optimized_prompt = ' '.join(optimized_words)
        
        # 简化句式结构
        replacements = {
            "请你帮我处理一下": "请处理",
            "我想了解一下": "请解释",
            "我需要知道": "请告诉我",
            "你能告诉我吗": "请告诉我",
            "你可以帮我吗": "请帮我",
            "非常感谢": "谢谢",
            "非常好": "很好",
            "我希望你能": "请",
            "我请求你": "请",
            "你能不能": "请",
            "你是否能够": "请",
            "麻烦你": "请",
            "感谢你的帮助": "谢谢",
            "我非常感激": "谢谢",
            "根据我的理解": "",
            "我认为": "",
            "我觉得": "",
            "我想": "",
            "应该是": "是",
            "可能是": "是",
            "大概是": "是",
            "基本上": "",
            "实际上": "",
            "事实上": "",
            "总的来说": "",
            "综上所述": "",
            "简而言之": ""
        }
        for old, new in replacements.items():
            optimized_prompt = optimized_prompt.replace(old, new)
        
        # 移除不必要的修饰词
        redundant_modifiers = ["非常", "极其", "特别", "十分", "相当", "非常", "无比", "异常"]
        for modifier in redundant_modifiers:
            optimized_prompt = optimized_prompt.replace(f"{modifier}", "")
        
        # 移除重复的标点符号
        import re
        optimized_prompt = re.sub(r'([.,!?;:])\1+', r'\1', optimized_prompt)
        
        # 修剪多余的空格
        optimized_prompt = re.sub(r'\s+', ' ', optimized_prompt).strip()
        
        return optimized_prompt
    
    def _get_dynamic_max_tokens(self, prompt_length: int, context_complexity: int) -> int:
        """根据提示词长度和上下文复杂度动态调整max_tokens参数
        
        Args:
            prompt_length: 提示词长度（字符数）
            context_complexity: 上下文复杂度（0-10）
        
        Returns:
            动态调整后的max_tokens值
        """
        # 基础token数
        base_tokens = 500
        
        # 根据提示词长度调整
        length_factor = min(prompt_length / 1000, 2.0)  # 最多增加2倍
        
        # 根据上下文复杂度调整
        complexity_factor = context_complexity / 5.0  # 0-2倍
        
        # 计算最终max_tokens，限制在合理范围内
        max_tokens = int(base_tokens * (1 + length_factor + complexity_factor))
        return max(500, min(3000, max_tokens))
    
    def _filter_response(self, response: str) -> str:
        """过滤响应内容，只保留核心信息，减少返回token数量
        
        Args:
            response: GPT-4o返回的原始响应
        
        Returns:
            过滤后的核心响应内容
        """
        # 移除开场白和结束语
        opening_phrases = ["根据上下文信息", "综上所述", "总结一下", "总的来说", "简而言之", "一言以蔽之", "概括来说"]
        closing_phrases = ["以上是我的回答", "希望对你有所帮助", "如有其他问题", "感谢你的提问", "谢谢"]
        
        for phrase in opening_phrases + closing_phrases:
            response = response.replace(phrase, "")
        
        # 保留关键信息，移除冗余内容
        lines = response.split('\n')
        filtered_lines = []
        
        # 定义关键信息标记词
        key_info_markers = ["建议", "结论", "结果", "原因", "解决方案", "步骤", "方法", "要点", "注意事项", "重要提示"]
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 移除过于冗长的解释
            if len(line) > 300:
                # 检查是否包含关键信息
                contains_key_info = any(marker in line for marker in key_info_markers)
                if contains_key_info:
                    # 保留关键部分
                    for marker in key_info_markers:
                        if marker in line:
                            parts = line.split(marker, 1)
                            if len(parts) > 1:
                                line = f"{marker}{parts[1]}"
                                break
                    # 限制长度
                    line = line[:300] + "..."
                else:
                    # 跳过非关键的冗长内容
                    continue
            
            # 移除重复的行
            if line not in filtered_lines:
                filtered_lines.append(line)
        
        # 移除重复的段落
        final_response = '\n'.join(filtered_lines)
        
        # 移除多余的空格和换行符
        import re
        final_response = re.sub(r'\s+', ' ', final_response)
        final_response = re.sub(r'\n+', '\n', final_response)
        final_response = final_response.strip()
        
        return final_response
    
    def process_with_gpt4o(self, encoding: UnifiedEncoding, prompt: str) -> str:
        """使用GPT-4o处理统一编码结果，优化API调用"""
        if not self.openai_client:
            logger.error("GPT-4o不可用，无法处理多模态输入")
            return "无法处理多模态输入，GPT-4o不可用"
        
        try:
            # 优化提示词
            optimized_prompt = self._optimize_prompt(prompt)
            
            messages = [
                {
                    "role": "system",
                    "content": "你是一个智能农业AI助手，能够理解多模态输入并提供专业的农业决策建议。请保持回答简洁，只提供核心信息。"
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": optimized_prompt}
                    ]
                }
            ]
            
            # 添加统一表示作为上下文（优化版本）
            if encoding.unified_representation:
                unified_repr = encoding.unified_representation
                # 优化上下文信息，移除冗余
                if len(unified_repr) > 1000:
                    unified_repr = unified_repr[:1000] + "..."  # 限制上下文长度
                messages[1]["content"].append(
                    {"type": "text", "text": f"上下文: {unified_repr}"}
                )
            
            # 计算上下文复杂度（简单估计）
            context_complexity = 0
            if encoding.text_encoding:
                context_complexity += 2
            if encoding.image_encodings and encoding.image_encodings.length > 0:
                context_complexity += 3
            if encoding.audio_encoding:
                context_complexity += 3
            context_complexity = min(context_complexity, 10)
            
            # 动态调整max_tokens
            dynamic_max_tokens = self._get_dynamic_max_tokens(len(optimized_prompt), context_complexity)
            
            # 调用GPT-4o API
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.7,
                max_tokens=dynamic_max_tokens
            )
            
            # 过滤响应内容
            filtered_response = self._filter_response(response.choices[0].message.content)
            
            return filtered_response
        except Exception as e:
            logger.error(f"GPT-4o处理失败: {str(e)}")
            return f"处理多模态输入时出错: {str(e)}"
    
    def get_domain_generalized_features(self, encoding: UnifiedEncoding, domain_id: int = 0) -> DomainGeneralizationResult:
        """获取领域泛化特征
        
        Args:
            encoding: 统一编码结果
            domain_id: 目标领域ID
            
        Returns:
            领域泛化结果
        """
        try:
            # 收集所有模态的特征
            all_features = []
            
            # 处理文本特征
            if encoding.text_encoding and isinstance(encoding.text_encoding.encoding, list):
                text_feat = torch.tensor(encoding.text_encoding.encoding).unsqueeze(0).unsqueeze(0)
                all_features.append(text_feat)
            
            # 处理图像特征
            for img_enc in encoding.image_encodings:
                if isinstance(img_enc.encoding, list):
                    img_feat = torch.tensor(img_enc.encoding).unsqueeze(0).unsqueeze(0)
                    all_features.append(img_feat)
            
            # 处理音频特征
            if encoding.audio_encoding and isinstance(encoding.audio_encoding.encoding, list):
                audio_feat = torch.tensor(encoding.audio_encoding.encoding).unsqueeze(0).unsqueeze(0)
                all_features.append(audio_feat)
            
            if not all_features:
                # 如果没有可用的嵌入向量，返回原始编码
                logger.warning("没有可用的嵌入向量进行领域泛化")
                return DomainGeneralizationResult(
                    original_features=torch.tensor([]),
                    generalized_features=torch.tensor([]),
                    domain_ids=torch.tensor([domain_id]),
                    confidence=0.5
                )
            
            # 拼接所有特征
            combined_features = torch.cat(all_features, dim=1)  # [batch_size, num_modalities, hidden_size]
            
            # 确保特征维度匹配Bi-ATEN模块的隐藏层大小
            batch_size, seq_len, hidden_size = combined_features.shape
            target_hidden_size = 256  # Bi-ATEN模块的默认隐藏层大小
            
            if hidden_size != target_hidden_size:
                # 调整特征维度
                projection = torch.nn.Linear(hidden_size, target_hidden_size)
                combined_features = projection(combined_features)
            
            # 生成领域ID张量
            domain_ids = torch.tensor([domain_id] * batch_size)
            
            # 使用Bi-ATEN模块获取领域泛化特征
            generalized_features = bi_aten_module.get_domain_generalized_features(combined_features, domain_ids)
            
            # 计算置信度（简单示例）
            confidence = 0.85
            
            return DomainGeneralizationResult(
                original_features=combined_features,
                generalized_features=generalized_features,
                domain_ids=domain_ids,
                confidence=confidence
            )
        except Exception as e:
            logger.error(f"领域泛化处理失败: {str(e)}")
            return DomainGeneralizationResult(
                original_features=torch.tensor([]),
                generalized_features=torch.tensor([]),
                domain_ids=torch.tensor([domain_id]),
                confidence=0.1
            )
    
    def encode_with_domain_generalization(self, input_data: MultimodalInput, target_domain_id: int = 0) -> Dict[str, Any]:
        """带领域泛化的多模态编码
        
        Args:
            input_data: 多模态输入数据
            target_domain_id: 目标领域ID
            
        Returns:
            包含原始编码和领域泛化结果的字典
        """
        # 进行常规编码
        original_encoding = self.encode_multimodal(input_data)
        
        # 获取领域泛化特征
        domain_result = self.get_domain_generalized_features(original_encoding, target_domain_id)
        
        return {
            "original_encoding": original_encoding,
            "domain_generalization": domain_result,
            "target_domain": target_domain_id,
            "timestamp": datetime.now()
        }


# 创建全局多模态编码器实例
multimodal_encoder = MultimodalEncoder()