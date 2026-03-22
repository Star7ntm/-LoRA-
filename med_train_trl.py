from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    TaskType,
    PeftModel
)
from datasets import load_dataset
import torch
from trl import SFTTrainer
import argparse
import json
from pathlib import Path

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="医疗模型训练脚本（优化版）")
    parser.add_argument(
        "--model_name",
        type=str,
        default="models/Qwen/Qwen3-1.7B",
        help="模型路径或HuggingFace模型ID"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="训练数据路径（JSONL格式）或数据集名称"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./sft_output",
        help="输出目录"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="最大训练样本数（用于快速测试）"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="最大训练步数（优先于num_train_epochs）"
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="训练轮数（如果未设置max_steps）"
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=2,
        help="每设备训练批次大小（8GB显存建议2-4）"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=16,
        help="梯度累积步数（8GB显存建议16-32）"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="学习率"
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="LoRA rank"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=128,
        help="最大序列长度（8GB显存建议128-256）"
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="保存checkpoint的步数间隔"
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=50,
        help="记录日志的步数间隔"
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Warmup比例"
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="从checkpoint恢复训练"
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        default=True,
        help="启用梯度检查点（节省显存，8GB显存建议启用）"
    )
    parser.add_argument(
        "--optim",
        type=str,
        default="adamw_8bit",
        help="优化器类型（adamw_8bit节省显存，adamw_torch更快）"
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="cosine",
        choices=["linear", "cosine", "constant"],
        help="学习率调度器类型（cosine更好收敛，linear更快）"
    )
    return parser.parse_args()

def load_data_from_jsonl(file_path, max_samples=None):
    """从JSONL文件加载数据"""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def main():
    # 清理显存缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    args = parse_args()
    
    # 自动调整参数（8GB显存优化）
    if torch.cuda.is_available():
        total_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if total_memory_gb < 10:  # 8GB显存
            # 自动启用梯度检查点
            if not args.gradient_checkpointing:
                args.gradient_checkpointing = True
                print(f"检测到显存 < 10GB，自动启用梯度检查点")
            # 自动调整批次大小
            if args.per_device_train_batch_size > 4:
                print(f"警告: 批次大小 {args.per_device_train_batch_size} 可能过大，建议 <= 4")
            # 自动调整序列长度
            if args.max_seq_length > 256:
                print(f"警告: 序列长度 {args.max_seq_length} 可能过大，建议 <= 256")
    
    print("="*60)
    print("医疗模型训练（优化版）")
    print("="*60)
    print(f"模型: {args.model_name}")
    print(f"输出目录: {args.output_dir}")
    if args.max_samples:
        print(f"最大样本数: {args.max_samples}")
    if args.max_steps:
        print(f"最大步数: {args.max_steps}")
    else:
        print(f"训练轮数: {args.num_train_epochs}")
    print(f"批次大小: {args.per_device_train_batch_size}")
    print(f"梯度累积: {args.gradient_accumulation_steps}")
    print(f"有效批次: {args.per_device_train_batch_size * args.gradient_accumulation_steps}")
    print("="*60)

    # 1. 加载模型和分词器
    print("\n加载模型和分词器...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        trust_remote_code=True,
        use_cache=False,  # 用于梯度检查点
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        padding_side="right", 
    )

    # 设置pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("模型加载完成")

    # 2. 配置LoRA
    print("\n配置LoRA...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 3. 如果指定了checkpoint，手动加载adapter权重（避免torch.load安全问题）
    checkpoint_path = None
    resume_step = 0
    if args.resume_from_checkpoint:
        checkpoint_path = args.resume_from_checkpoint
        print(f"\n从checkpoint加载模型权重: {checkpoint_path}")
        try:
            # 读取checkpoint的trainer_state.json获取当前步数
            trainer_state_path = Path(checkpoint_path) / "trainer_state.json"
            if trainer_state_path.exists():
                with open(trainer_state_path, "r", encoding="utf-8") as f:
                    trainer_state = json.load(f)
                    resume_step = trainer_state.get("global_step", 0)
                    print(f"检测到checkpoint步数: {resume_step}")
            
            # 使用PeftModel的load_adapter方法加载权重（使用safetensors，避免torch.load安全问题）
            from peft import PeftModel
            if isinstance(model, PeftModel):
                model.load_adapter(checkpoint_path, adapter_name="default")
                print("模型权重加载成功（优化器和调度器将从头开始）")
            else:
                # 如果不是PeftModel，尝试直接加载
                model.load_adapter(checkpoint_path)
                print("模型权重加载成功（优化器和调度器将从头开始）")
        except Exception as e:
            print(f"警告: 无法加载checkpoint权重: {e}")
            print("将从头开始训练")
            checkpoint_path = None
            resume_step = 0

    # 4. 启用梯度检查点（8GB显存建议启用）
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("已启用梯度检查点（节省显存，适合8GB显存）")
    else:
        print("未启用梯度检查点（更快速度，需要更多显存）")

    # 4. 加载数据集
    print("\n加载数据集...")
    if args.data_path:
        if args.data_path.endswith(".jsonl"):
            # 从JSONL文件加载
            print(f"从JSONL文件加载: {args.data_path}")
            data = load_data_from_jsonl(args.data_path, args.max_samples)
            print(f"已加载 {len(data)} 条数据")
            
            # 转换为datasets格式
            def format_dataset(example):
                """将数据转换为Qwen3的对话格式"""
                messages = [
                    {"role": "user", "content": example["query"]},
                    {"role": "assistant", "content": example["response"]},
                ]
                
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                # 如果文本太长，进行截断
                tokens = tokenizer.encode(text, add_special_tokens=False)
                if len(tokens) > args.max_seq_length:
                    tokens = tokens[:args.max_seq_length]
                    text = tokenizer.decode(tokens, skip_special_tokens=True)
                return {"text": text}
            
            # 创建数据集
            from datasets import Dataset
            dataset = Dataset.from_list(data)
            dataset = dataset.map(format_dataset, remove_columns=["query", "response"])
        else:
            # 从HuggingFace数据集加载
            print(f"从HuggingFace数据集加载: {args.data_path}")
            dataset = load_dataset("json", data_files=args.data_path)
            
            def format_dataset(example):
                messages = [
                    {"role": "user", "content": example["query"]},
                    {"role": "assistant", "content": example["response"]},
                ]
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                # 如果文本太长，进行截断
                tokens = tokenizer.encode(text, add_special_tokens=False)
                if len(tokens) > args.max_seq_length:
                    tokens = tokens[:args.max_seq_length]
                    text = tokenizer.decode(tokens, skip_special_tokens=True)
                return {"text": text}
            
            dataset = dataset.map(format_dataset)
            if "train" in dataset:
                dataset = dataset["train"]
    else:
        # 默认数据集
        print("使用默认数据集")
        dataset = load_dataset("my_datasets/michaelwzhu/ShenNong_TCM_Dataset")
        
        def format_dataset(example):
            messages = [
                {"role": "user", "content": example["query"]},
                {"role": "assistant", "content": example["response"]},
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            # 如果文本太长，进行截断
            tokens = tokenizer.encode(text, add_special_tokens=False)
            if len(tokens) > args.max_seq_length:
                tokens = tokens[:args.max_seq_length]
                text = tokenizer.decode(tokens, skip_special_tokens=True)
            return {"text": text}
        
        dataset = dataset.map(format_dataset)
        if "train" in dataset:
            dataset = dataset["train"]
    
    # 限制数据量
    if args.max_samples and len(dataset) > args.max_samples:
        print(f"警告: 限制数据量: {len(dataset)} -> {args.max_samples}")
        dataset = dataset.select(range(args.max_samples))
    
    print(f"数据集加载完成: {len(dataset)} 条数据")

    # 5. 配置训练参数
    print("\n配置训练参数...")
    # 如果设置了max_steps，num_train_epochs应该设置为一个很大的值，而不是None
    # 这样TrainingArguments内部不会报错
    num_epochs = args.num_train_epochs
    max_steps = args.max_steps
    if args.max_steps and resume_step > 0:
        # 如果从checkpoint恢复，max_steps表示总步数，需要调整
        # 但由于trainer.train()会从头开始计数，我们需要保持max_steps不变
        # 实际训练会从0计数到max_steps，但模型权重已经是从resume_step加载的
        print(f"注意: 从checkpoint恢复，当前步数: {resume_step}, 目标总步数: {max_steps}")
        if max_steps <= resume_step:
            print(f"警告: max_steps ({max_steps}) <= resume_step ({resume_step})")
            print(f"将训练到 {max_steps} 步（总步数）")
    if args.max_steps:
        # 当使用max_steps时，设置一个很大的epoch数，实际训练会在max_steps时停止
        num_epochs = 1000
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=num_epochs,
        max_steps=max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim=args.optim,  # 优化器类型
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,  # 学习率调度器类型
        gradient_checkpointing=args.gradient_checkpointing,
        remove_unused_columns=False,
        dataloader_pin_memory=True,  # 启用pin_memory加速数据加载
        dataloader_num_workers=0,  # Windows上设为0避免问题
        report_to=[],  # 禁用wandb等记录
        save_total_limit=2,  # 减少保存的checkpoint数量
        save_safetensors=True,  # 使用safetensors格式
        ddp_find_unused_parameters=False,  # 加速分布式训练（如果使用）
        group_by_length=False,  # 禁用按长度分组，加快数据加载
    )

    # 6. 创建训练器
    print("\n创建训练器...")
    # 注意：SFTTrainer 在某些版本中不支持 max_seq_length 参数
    # 序列长度限制通过 tokenizer 的 truncation 在数据预处理时处理
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        processing_class=tokenizer,
        # max_seq_length 参数在某些 TRL 版本中不支持，已移除
        # 序列长度限制在 format_dataset 函数中通过 tokenizer 处理
    )

    # 7. 计算预计训练时间
    total_steps = max_steps if max_steps else (
        len(dataset) // (args.per_device_train_batch_size * args.gradient_accumulation_steps) * args.num_train_epochs
    )
    print(f"\n训练信息:")
    print(f"   数据量: {len(dataset)} 条")
    if resume_step > 0:
        print(f"   从checkpoint恢复: 步数 {resume_step}")
        print(f"   目标总步数: {total_steps}")
        print(f"   将训练: {total_steps} 步（模型权重已从步数 {resume_step} 加载）")
    else:
        print(f"   预计总步数: {total_steps}")
    print(f"   有效批次大小: {args.per_device_train_batch_size * args.gradient_accumulation_steps}")
    print(f"   每步样本数: {args.per_device_train_batch_size * args.gradient_accumulation_steps}")
    print(f"   预计训练时间: 约 {total_steps * 2 / 60:.1f} 分钟（假设每步2秒）")

    # 8. 开始训练
    print("\n开始训练...")
    # 如果已经手动加载了checkpoint权重，就不传递resume_from_checkpoint给trainer.train()
    # 这样可以避免torch.load的安全问题，优化器和调度器会从头开始
    if checkpoint_path:
        print("注意: 已手动加载模型权重，优化器和调度器将从头开始训练")
    trainer.train()

    # 9. 保存模型
    print("\n保存模型...")
    trainer.save_model(f"{args.output_dir}/final_model")
    print(f"模型已保存到: {args.output_dir}/final_model")

if __name__ == "__main__":
    main()
