import json
import time
from typing import List, Optional, Union
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TextIteratorStreamer
from threading import Thread
import uvicorn
from contextlib import asynccontextmanager
import logging
import argparse
from peft import PeftModel

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 默认模型配置（如果未通过参数指定）
DEFAULT_MODEL_NAME = "models/Qwen/Qwen3-1.7B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_MAP = "auto" if torch.cuda.is_available() else None

def get_optimal_max_length():
    """根据显存自动选择最大上下文长度"""
    if not torch.cuda.is_available():
        return 1024
    
    total_memory = torch.cuda.get_device_properties(0).total_memory
    if total_memory < 8e9:  # < 8GB
        return 1024
    elif total_memory < 16e9:  # < 16GB
        return 2048
    else:  # >= 16GB
        return 4096

DEFAULT_MAX_LENGTH = get_optimal_max_length()

# 全局变量
model = None
tokenizer = None
current_model_name = None
lora_checkpoint = None

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="启动Qwen API服务器")
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"模型名称或路径 (默认: {DEFAULT_MODEL_NAME})"
    )
    parser.add_argument(
        "--lora-checkpoint",
        type=str,
        default=None,
        help="LoRA checkpoint路径（可选，用于加载训练后的LoRA权重）"
    )
    return parser.parse_args()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理
    """
    global model, tokenizer, current_model_name, lora_checkpoint
    
    # 启动时加载模型
    logger.info(f"正在加载模型: {current_model_name}...")
    try:
        # 加载tokenizer（优先从checkpoint加载，否则从基础模型加载）
        if lora_checkpoint:
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    lora_checkpoint,
                    trust_remote_code=True
                )
                logger.info(f"从LoRA checkpoint加载tokenizer: {lora_checkpoint}")
            except Exception as e:
                logger.warning(f"从checkpoint加载tokenizer失败，使用基础模型: {e}")
                tokenizer = AutoTokenizer.from_pretrained(
                    current_model_name,
                    trust_remote_code=True
                )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                current_model_name,
                trust_remote_code=True
            )
        
        # 加载基础模型
        model = AutoModelForCausalLM.from_pretrained(
            current_model_name,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=DEVICE_MAP,
            trust_remote_code=True
        )
        
        # 如果指定了LoRA checkpoint，加载LoRA权重
        if lora_checkpoint:
            logger.info(f"正在加载LoRA权重: {lora_checkpoint}...")
            try:
                model = PeftModel.from_pretrained(model, lora_checkpoint)
                logger.info(f"LoRA权重加载成功")
            except Exception as e:
                logger.error(f"LoRA权重加载失败: {str(e)}")
                raise e
        
        # 设置为评估模式
        model.eval()
        
        logger.info(f"模型加载完成，设备: {model.device}")
        logger.info(f"使用模型: {current_model_name}")
        if lora_checkpoint:
            logger.info(f"LoRA checkpoint: {lora_checkpoint}")
        logger.info(f"自动检测的最大上下文长度: {DEFAULT_MAX_LENGTH}")
        if torch.cuda.is_available():
            total_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"GPU显存: {total_memory_gb:.2f} GB")
        
    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}")
        raise e
    yield
    # 关闭时清理
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# 创建FastAPI应用
app = FastAPI(
    title="Qwen API Server",
    description="支持OpenAI兼容接口的Qwen模型API服务器",
    version="1.0.0"
)

# 添加CORS中间件（允许跨域请求）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源（生产环境应限制为特定域名）
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有HTTP方法
    allow_headers=["*"],  # 允许所有请求头
)

# 添加请求验证错误处理
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """处理请求验证错误"""
    logger.error(f"请求验证失败: {exc.errors()}")
    logger.error(f"请求体: {await request.body()}")
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "message": f"请求验证失败: {exc.errors()}",
                "type": "validation_error",
                "details": exc.errors()
            }
        }
)

# Pydantic模型定义
class ChatMessage(BaseModel):
    role: str = Field(..., description="消息角色：system, user, assistant")
    content: str = Field(..., description="消息内容")

class ChatCompletionRequest(BaseModel):
    model: str = Field(default="Qwen3", description="模型名称")
    messages: List[ChatMessage] = Field(..., description="消息列表")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="温度参数")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="核采样参数")
    max_tokens: int = Field(default=DEFAULT_MAX_LENGTH, ge=1, description="最大生成token数")
    stream: bool = Field(default=False, description="是否流式输出")
    stop: Optional[Union[str, List[str]]] = Field(default=None, description="停止词")
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="频率惩罚")
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="存在惩罚")

class ModelInfo(BaseModel):
    model_name: str
    device: str
    status: str

def format_messages(messages: List[ChatMessage]) -> str:
    """
    将消息列表格式化为Qwen模型所需的格式
    优先使用tokenizer的chat template
    """
    # 转换为字典格式
    messages_dict = [{"role": msg.role, "content": msg.content} for msg in messages]
    
    # 尝试使用tokenizer的chat template
    try:
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
            formatted_text = tokenizer.apply_chat_template(
                messages_dict,
                tokenize=False,
                add_generation_prompt=True
            )
            return formatted_text
    except Exception as e:
        logger.warning(f"使用chat template失败，回退到手动格式化: {e}")
    
    # 回退到手动格式化
    formatted_text = ""
    for msg in messages:
        if msg.role == "system":
            formatted_text += f"<|im_start|>system\n{msg.content}<|im_end|>\n"
        elif msg.role == "user":
            formatted_text += f"<|im_start|>user\n{msg.content}<|im_end|>\n"
        elif msg.role == "assistant":
            formatted_text += f"<|im_start|>assistant\n{msg.content}<|im_end|>\n"
    # 添加助手开始的标记
    formatted_text += "<|im_start|>assistant\n"
    return formatted_text

def generate_stream_chat_completion(
    messages: List[ChatMessage],
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = DEFAULT_MAX_LENGTH,
    stop: Optional[Union[str, List[str]]] = None
):
    """
    生成流式聊天完成
    """
    # 检查模型和tokenizer是否已加载
    if model is None or tokenizer is None:
        logger.error(f"Model is None: {model is None}, Tokenizer is None: {tokenizer is None}")
        raise ValueError("模型或tokenizer未加载")
    logger.info(f"开始生成，模型设备: {model.device}")
    logger.info(f"收到消息数量: {len(messages)}")
    # 格式化消息
    prompt = format_messages(messages)
    logger.info(f"格式化后的提示长度: {len(prompt)}")
    # 编码输入
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    logger.info(f"输入tensor形状: {inputs.input_ids.shape}")
    
    # 创建流式生成器
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        timeout=60.0
    )
    logger.info(f"Streamer创建完成: {streamer}")
    
    # 生成参数
    generation_config = {
        "max_new_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "streamer": streamer
    }
    logger.info(f"生成配置: {generation_config}")
    # 停止词处理
    if stop:
        if isinstance(stop, str):
            stop = [stop]
        stop_token_ids = []
        for s in stop:
            try:
                encoded = tokenizer.encode(s, add_special_tokens=False)
                if encoded:
                    stop_token_ids.append(encoded[0])
            except Exception as e:
                logger.warning(f"编码停止词失败 {s}: {e}")
        if stop_token_ids:
            generation_config["eos_token_id"] = stop_token_ids + [tokenizer.eos_token_id]
    
    # 在单独的线程中生成
    thread = Thread(target=model.generate, kwargs={
        "input_ids": inputs.input_ids,
        **generation_config
    })
    thread.start()
    logger.info(f"生成线程已启动")
    # 流式返回
    try:
        for text in streamer:
            if text:
                # 移除推理标记
                text = text.replace("<think>", "").replace("</think>", "")
                text = text.replace("<think>", "").replace("</think>", "")
                # 移除可能的停止词
                if stop:
                    for stop_word in stop:
                        text = text.replace(stop_word, "")
                if text.strip():  # 只返回非空内容
                    yield text
        logger.info("流式生成完成")
    except Exception as e:
        logger.error(f"流式迭代错误: {str(e)}", exc_info=True)
        raise

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """创建聊天完成（OpenAI兼容接口）- 仅流式输出"""
    try:
        logger.info(f"收到请求: model={request.model}, messages数量={len(request.messages)}")
        # 验证模型
        if "qwen" not in request.model.lower():
            logger.warning(f"请求模型 {request.model} 不匹配，使用默认模型")
        # 检查模型和tokenizer
        if model is None:
            raise HTTPException(status_code=503, detail="模型未加载")
        if tokenizer is None:
            raise HTTPException(status_code=503, detail="Tokenizer未加载")
    except Exception as e:
        logger.error(f"请求处理错误: {str(e)}", exc_info=True)
        raise HTTPException(status_code=422, detail=f"请求格式错误: {str(e)}")
    # 生成响应ID和时间戳
    response_id = f"chatcmpl-{int(time.time() * 1000)}"
    created = int(time.time())
    # 流式响应生成器
    def stream_generator():
        full_response = ""
        try:
            for chunk in generate_stream_chat_completion(
                messages=request.messages,
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
                stop=request.stop
            ):
                full_response += chunk
                
                # 构建流式响应块
                chunk_data = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": request.model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": chunk},
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"
            # 发送完成标记
            end_chunk = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            }
            yield f"data: {json.dumps(end_chunk, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"流式生成错误: {str(e)}", exc_info=True)
            error_chunk = {
                "error": {
                    "message": str(e),
                    "type": "generation_error"
                }
            }
            yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream"
    )

if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()
    # 设置当前模型名称和LoRA checkpoint
    current_model_name = args.model_name
    lora_checkpoint = args.lora_checkpoint
    # 设置应用的生命周期管理
    app.router.lifespan_context = lifespan
    # 启动服务器
    logger.info(f"启动服务器，模型: {current_model_name}")
    if lora_checkpoint:
        logger.info(f"LoRA checkpoint: {lora_checkpoint}")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1
    )