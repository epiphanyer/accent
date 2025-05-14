from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
import torch
import librosa
import numpy as np
import io

# 初始化 FastAPI 应用
app = FastAPI()

# 加载预训练模型（示例：联合处理音频和文本的模型）
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import (
    PtEngine, RequestConfig, safe_snapshot_download, get_model_tokenizer, get_template, InferRequest
)
from swift.tuners import Swift
# 请调整下面几行
model = 'Qwen/Qwen2-Audio-7B-Instruct'
# lora_checkpoint = safe_snapshot_download('output/v6-20250513-095250/checkpoint-878')  # 修改成checkpoint_dir
# model = 'atisto/accent'
lora_checkpoint = safe_snapshot_download('atisto/accent')
template_type = None  # None: 使用对应模型默认的template_type
default_system = "你是一个有用无害的助手。"  # None: 使用对应模型默认的default_system

# 加载模型和对话模板
model, tokenizer = get_model_tokenizer(model)
model = Swift.from_pretrained(model, lora_checkpoint)
template_type = template_type or model.model_meta.template
template = get_template(template_type, tokenizer, default_system=default_system)
engine = PtEngine.from_model_template(model, template, max_batch_size=2)
request_config = RequestConfig(max_tokens=512, temperature=0)

def translate(audio):
    infer_requests = [
        InferRequest(messages=[{'role': 'user', 'content': '将下面一段邵阳话方言音频转录为汉字:<audio>'}],
                     audios=[audio]),
    ]
    resp_list = engine.infer(infer_requests, request_config)
    return resp_list[0].choices[0].message.content

# 定义输入数据模型
class RequestData(BaseModel):
    text: str
    audio: UploadFile = File(...)  # 接收上传的音频文件

# 定义 API 路由
@app.post("/predict")
async def predict_joint(text: str = Form(...), audio: UploadFile = File(...)):
    # 验证文件类型
    # if not audio.content_type.startswith("audio/"):
    #     raise HTTPException(status_code=400, detail=f"{audio.content_type},Invalid file type. Only audio files are allowed.")
    
    # 读取音频文件内容
    audio_bytes = await audio.read()
    # 调用模型推理
    result = translate(audio_bytes)
    return {"result": result}

# 启动服务
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)