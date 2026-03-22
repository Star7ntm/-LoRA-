"""
使用vLLM加速推理的API服务器
安装: pip install vllm
"""
import json
import time
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
import uvicorn
import argparse
import logging
from contextlib import asynccontextmanager

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 默认模型配置
DEFAULT_MODEL_NAME = "./models/Qwen/Qwen3-1.7B"
DEFAULT_MAX_MODEL_LEN = 2048
DEFAULT_GPU_MEMORY_UTILIZATION = 0.9

# 全局变量
llm_engine = None
tokenizer = None

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="启动vLLM API服务器")
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"模型名称或路径 (默认: {DEFAULT_MODEL_NAME})"
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=DEFAULT_MAX_MODEL_LEN,
        help=f"最大模型长度 (默认: {DEFAULT_MAX_MODEL_LEN})"
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=DEFAULT_GPU_MEMORY_UTILIZATION,
        help=f"GPU内存利用率 (默认: {DEFAULT_GPU_MEMORY_UTILIZATION})"
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="张量并行大小 (默认: 1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="服务器端口 (默认: 8000)"
    )
    return parser.parse_args()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理
    """
    global llm_engine, tokenizer
    
    # 启动时初始化vLLM引擎
    logger.info(f"正在初始化vLLM引擎: {args.model_name}...")
    try:
        engine_args = AsyncEngineArgs(
            model=args.model_name,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            trust_remote_code=True,
        )
        llm_engine = AsyncLLMEngine.from_engine_args(engine_args)
        
        # 加载tokenizer用于格式化
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            trust_remote_code=True
        )
        
        logger.info(f"vLLM引擎初始化完成")
        logger.info(f"模型: {args.model_name}")
        logger.info(f"最大长度: {args.max_model_len}")
        logger.info(f"GPU内存利用率: {args.gpu_memory_utilization}")
        
    except Exception as e:
        logger.error(f"vLLM引擎初始化失败: {str(e)}")
        raise e
    yield
    # 关闭时清理
    if llm_engine:
        del llm_engine

# 创建FastAPI应用
app = FastAPI(
    title="vLLM API Server",
    description="使用vLLM加速推理的API服务器",
    version="1.0.0"
)

# 请求模型
class ChatMessage(BaseModel):
    role: str = Field(..., description="消息角色: system, user, assistant")
    content: str = Field(..., description="消息内容")

class ChatCompletionRequest(BaseModel):
    model: str = Field(default="default", description="模型名称")
    messages: List[ChatMessage] = Field(..., description="对话消息列表")
    temperature: float = Field(default=0.7, ge=0, le=2, description="采样温度")
    top_p: float = Field(default=0.9, ge=0, le=1, description="Top-p采样")
    max_tokens: int = Field(default=2048, ge=1, le=4096, description="最大生成token数")
    stream: bool = Field(default=False, description="是否流式返回")

def format_messages(messages: List[ChatMessage]) -> str:
    """格式化消息为模型输入"""
    messages_dict = [{"role": msg.role, "content": msg.content} for msg in messages]
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
    formatted = ""
    for msg in messages:
        if msg.role == "system":
            formatted += f"系统: {msg.content}\n\n"
        elif msg.role == "user":
            formatted += f"用户: {msg.content}\n\n"
        elif msg.role == "assistant":
            formatted += f"助手: {msg.content}\n\n"
    formatted += "助手: "
    return formatted

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Chat completions端点"""
    try:
        # 格式化输入
        prompt = format_messages(request.messages)
        
        # 设置采样参数
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            stop=None
        )
        
        if request.stream:
            # 流式返回
            async def generate_stream():
                async for request_output in llm_engine.generate(
                    prompt, sampling_params, request_id=f"req_{int(time.time())}"
                ):
                    if request_output.finished:
                        break
                    token = request_output.outputs[0].token_ids[-1]
                    text = tokenizer.decode([token], skip_special_tokens=True)
                    yield f"data: {json.dumps({'choices': [{'delta': {'content': text}}]}, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
            
            return StreamingResponse(generate_stream(), media_type="text/event-stream")
        else:
            # 非流式返回
            request_id = f"req_{int(time.time())}"
            async for request_output in llm_engine.generate(
                prompt, sampling_params, request_id=request_id
            ):
                if request_output.finished:
                    generated_text = request_output.outputs[0].text
                    return JSONResponse(content={
                        "choices": [{
                            "message": {
                                "role": "assistant",
                                "content": generated_text
                            }
                        }]
                    })
            
            return JSONResponse(content={"error": "生成失败"})
            
    except Exception as e:
        logger.error(f"推理错误: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "ok", "engine": "vLLM"}

if __name__ == "__main__":
    args = parse_args()
    uvicorn.run(app, host="0.0.0.0", port=args.port)


