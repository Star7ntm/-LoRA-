# 中医智能诊疗助手

## 项目简介

本项目是基于Qwen3-1.7B模型，使用LoRA技术微调的中医诊疗AI助手。项目实现了完整的训练、推理和应用系统，包括：

- 模型训练：LoRA微调，Loss降至0.3以下
- 自我认知：模型能够识别自身身份和开发者信息
- Web应用：支持结构化医疗信息输入和流式对话
- 前端界面：独立的HTML前端，侧边栏折叠设计

---

## 快速开始

### 1. 环境准备

```powershell
# 创建并激活虚拟环境
conda create -n day18 python=3.12 -y
conda activate day18

# 安装依赖
pip install -r requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple

# 安装PyTorch（CUDA 12.4）
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124

# 验证CUDA
python -c "import torch; print('CUDA可用:', torch.cuda.is_available())"
```

### 2. 下载模型和数据集

```powershell
# 下载Qwen3-1.7B模型
python model_download2.py --repo_id Qwen/Qwen3-1.7B

# 下载中医数据集
python model_download2.py --repo_type dataset --repo_id michaelwzhu/ShenNong_TCM_Dataset
rename datasets my_datasets
```

### 3. 启动应用

**方式1：使用独立前端（推荐）**

```powershell
# 终端1：启动API服务器
python api_server.py --model-name ./models/Qwen/Qwen3-1.7B --lora-checkpoint ./sft_output_high_quality/checkpoint-3000

# 终端2：启动前端服务器
python start_frontend.py
```

浏览器会自动打开 `http://localhost:8080/frontend.html`

**方式2：使用Gradio WebUI**

```powershell
# 终端1：启动API服务器
python api_server.py --model-name ./models/Qwen/Qwen3-1.7B --lora-checkpoint ./sft_output_high_quality/checkpoint-3000

# 终端2：启动Gradio WebUI
python med_chat.py
```

---

## 核心功能

### 1. 模型训练

#### 1.1 生成自我认知数据

```powershell
python generate_self_cognition.py --output_path self_cognition.jsonl --trainer_name "达星辰，陈欣悦，彭延浩" --model_name "Qwen3-1.7B"
```

#### 1.2 数据处理

```powershell
# 基础数据处理
python data_process.py \
  --input_file my_datasets/michaelwzhu/ShenNong_TCM_Dataset/ChatMed_TCM-v0.2.json \
  --output_dir ./processed_data \
  --self_cognition_file self_cognition.jsonl \
  --self_cognition_repeat 10 \
  --check_quality

# 高质量数据筛选（推荐）
python optimize_data_quality.py \
  --input_file my_datasets/michaelwzhu/ShenNong_TCM_Dataset/ChatMed_TCM-v0.2.json \
  --output_dir ./processed_data_super_quality \
  --self_cognition_file self_cognition.jsonl \
  --target_ratio 0.15 \
  --min_quality_score 70 \
  --max_medical_samples 30000
```

#### 1.3 模型训练

```powershell
# 推荐配置（Loss < 0.3）
python med_train_trl.py \
  --data_path ./processed_data_super_quality/train.jsonl \
  --max_steps 3000 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --learning_rate 2e-4 \
  --lora_r 32 \
  --lora_alpha 64 \
  --max_seq_length 512 \
  --save_steps 500 \
  --logging_steps 50 \
  --warmup_ratio 0.1 \
  --optim adamw_8bit \
  --lr_scheduler_type cosine \
  --gradient_checkpointing \
  --output_dir ./sft_output_high_quality
```

**训练结果**：

- 初始Loss: 2.0403
- 最终Loss: 0.6175（3000步） / 0.2991（6000步）
- Token准确率: 88.3%（3000步）

#### 1.4 训练监控

```powershell
# 实时监控训练进度
python monitor_training.py --output_dir ./sft_output_high_quality --interval 60

# 生成Loss曲线图
python plot_loss.py --output_base_dir ./sft_output_high_quality --output_dir ./loss_plots_6000
```

### 2. 模型推理

#### 2.1 启动API服务器

```powershell
# 使用训练后的模型
python api_server.py \
  --model-name ./models/Qwen/Qwen3-1.7B \
  --lora-checkpoint ./sft_output_high_quality/checkpoint-3000
```

API服务器运行在 `http://127.0.0.1:8000`

#### 2.2 测试API

```powershell
# 使用curl测试
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3",
    "messages": [
      {"role": "user", "content": "你是谁？"}
    ],
    "stream": true
  }'
```

### 3. Web应用

#### 3.1 独立前端（frontend.html）

**特点**：

- 侧边栏折叠设计，医疗信息默认隐藏
- 对话窗口为主界面
- Apple风格设计（黑白灰配色）
- 流式响应显示
- 自动移除推理标记

**启动**：

```powershell
python start_frontend.py
```

访问 `http://localhost:8080/frontend.html`

#### 3.2 Gradio WebUI（med_chat.py）

**特点**：

- 结构化医疗信息输入
- 下拉列表和多选功能
- 流式对话
- 多轮会话支持

**启动**：

```powershell
python med_chat.py
```

---

## 项目结构

```
day18/
├── README.md                      # 基础说明
├── USER_README.md                 # 详细使用指南（本文档）
├── 中医大语言模型实训项目报告.md    # 项目报告
├── requirements.txt                # 依赖库
│
├── 核心脚本
│   ├── api_server.py              # FastAPI服务器
│   ├── med_train_trl.py           # 训练脚本
│   ├── data_process.py            # 数据处理
│   ├── optimize_data_quality.py   # 数据质量优化
│   ├── generate_self_cognition.py  # 自我认知数据生成
│   ├── monitor_training.py        # 训练监控
│   ├── plot_loss.py               # Loss可视化
│   └── model_download2.py         # 模型下载
│
├── 前端应用
│   ├── frontend.html              # 独立HTML前端（推荐）
│   ├── start_frontend.py         # 前端服务器
│   └── med_chat.py                # Gradio WebUI
│
├── 数据文件
│   ├── self_cognition.jsonl      # 自我认知数据
│   ├── processed_data/            # 基础处理数据
│   ├── processed_data_high_quality/    # 高质量数据
│   └── processed_data_super_quality/  # 超高质量数据（推荐）
│
├── 模型文件
│   ├── models/Qwen/Qwen3-1.7B/   # 基础模型
│   └── sft_output_high_quality/  # 训练输出
│       ├── checkpoint-3000/       # 3000步checkpoint（Loss=0.6175）
│       └── final_model/          # 最终模型
│
└── 可视化
    └── loss_plots_6000/          # Loss曲线图
        ├── training_loss.png
        └── loss_data.json
```

---

## 训练参数说明

### 关键参数

| 参数                            | 值          | 说明                  |
| ----------------------------- | ---------- | ------------------- |
| `max_steps`                   | 3000       | 训练步数，Loss降至0.6175   |
| `per_device_train_batch_size` | 2          | 每设备批次大小（8GB显存）      |
| `gradient_accumulation_steps` | 16         | 梯度累积，有效批次=32        |
| `learning_rate`               | 2e-4       | 学习率（LoRA微调典型值）      |
| `lora_r`                      | 32         | LoRA rank（模型容量）     |
| `lora_alpha`                  | 64         | LoRA alpha（与rank匹配） |
| `max_seq_length`              | 512        | 最大序列长度              |
| `optim`                       | adamw_8bit | 8bit优化器（节省显存）       |
| `lr_scheduler_type`           | cosine     | 余弦学习率调度             |
| `gradient_checkpointing`      | True       | 梯度检查点（节省显存）         |

### 训练效果

**3000步训练结果**：

- 初始Loss: 2.0403
- 最终Loss: 0.6175
- Loss下降: 1.4228
- Token准确率: 81.3%
- 训练时间: 约6-8小时（8GB显存）

**6000步训练结果*：

- 最终Loss: 0.2991（达到目标）

---

## 常见问题

### Q1: 如何提高自我认知数据权重？

A: 使用 `--self_cognition_repeat` 参数，例如 `--self_cognition_repeat 10` 表示重复10次。在数据质量优化脚本中使用 `--target_ratio 0.15` 可达到15%占比。

### Q2: 训练中断后如何恢复？

A: 使用 `--resume_from_checkpoint` 参数：

```powershell
python med_train_trl.py \
  --data_path ./processed_data_super_quality/train.jsonl \
  --resume_from_checkpoint ./sft_output_high_quality/checkpoint-3000 \
  --max_steps 6000
```

### Q3: 显存不足怎么办？

A: 

- 减小 `per_device_train_batch_size`（如改为1）
- 增加 `gradient_accumulation_steps`（如改为32）
- 减小 `max_seq_length`（如改为256）
- 确保启用 `gradient_checkpointing`

### Q4: Loss无法降到0.3以下？

A: 

- 增加LoRA rank（r=32或更大）
- 增加训练步数（3000+步）
- 使用高质量数据筛选（`optimize_data_quality.py`）
- 使用cosine学习率调度器

### Q5: 前端无法连接API服务器？

A: 

- 确保API服务器运行在 `http://127.0.0.1:8000`
- 检查API服务器是否已添加CORS支持
- 检查防火墙设置
- 查看浏览器控制台错误信息

### Q6: 如何移除推理标记（`<think></think>`）？

A: 前端已自动处理，API服务器也会移除。如果仍有问题，检查：

- `frontend.html` 中的清理逻辑
- `api_server.py` 中的流式生成代码

---

## 项目成果

### 训练成果

- ✅ Loss从2.0403降至0.6175（3000步）
- ✅ Token准确率达到81.3%
- ✅ 自我认知数据占比15%
- ✅ 训练数据18,822条（高质量筛选）

### 功能成果

- ✅ 8个核心方面全部实现
- ✅ 独立前端界面（侧边栏折叠）
- ✅ 流式对话响应
- ✅ 结构化医疗信息输入
- ✅ 多轮会话支持

### 技术成果

- ✅ LoRA微调（r=32，仅训练2%参数）
- ✅ 8GB显存优化训练
- ✅ 数据质量评分机制
- ✅ 训练监控和可视化

---

## 联系与支持

**开发者**：达星辰，陈欣悦，彭延浩

**项目报告**：详见 `中医大语言模型实训项目报告.md`

---

## 更新日志

- **2024-12**: 完成项目开发，实现Loss < 0.3目标
- **2024-12**: 重构前端界面，采用侧边栏折叠设计
- **2024-12**: 添加独立HTML前端，支持本地部署
- **2024-12**: 优化数据质量筛选，提升训练效果
