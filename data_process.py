import json
import random
from pathlib import Path
from generate_self_cognition import generate_self_cognition_data

def load_data(input_file):
    """加载数据文件（支持JSON和JSONL格式）"""
    data = []
    
    with open(input_file, "r", encoding="utf-8") as f:
        # 先尝试作为JSONL加载
        try:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
            if data:
                return data
        except:
            pass
        
        # 如果不是JSONL，尝试作为标准JSON加载
        f.seek(0)
        try:
            data = json.load(f)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                # 如果是字典，尝试找到数据数组
                for key in ["data", "items", "samples"]:
                    if key in data and isinstance(data[key], list):
                        return data[key]
                return [data]
        except:
            pass
    
    return []

def process_medical_data(input_file, output_dir="./processed_data", trainer_name="达星辰，陈欣悦，彭延浩", model_name="Qwen3-1.7B", self_cognition_repeat=None, self_cognition_file="self_cognition.jsonl", target_ratio=0.08):
    """处理医疗数据，合并自我认知数据
    
    Args:
        input_file: 输入数据文件路径
        output_dir: 输出目录
        trainer_name: 训练者姓名
        model_name: 模型名称
        self_cognition_repeat: 自我认知数据重复次数（如果为None，则根据target_ratio自动计算）
        self_cognition_file: 自我认知数据文件路径
        target_ratio: 目标自我认知数据占比（默认8%，即0.08）
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    print(f"加载数据文件: {input_file}")
    # 1. 加载原始数据
    raw_data = load_data(input_file)
    print(f"   原始数据量: {len(raw_data)} 条")
    
    # 2. 加载或生成自我认知数据
    self_cognition_path = Path(self_cognition_file)
    if self_cognition_path.exists():
        print(f"从文件加载自我认知数据: {self_cognition_file}")
        self_cognition = load_data(self_cognition_file)
        print(f"   自我认知数据: {len(self_cognition)} 条")
    else:
        print(f"生成自我认知数据（文件 {self_cognition_file} 不存在）...")
        self_cognition = generate_self_cognition_data(trainer_name, model_name)
        print(f"   自我认知数据: {len(self_cognition)} 条")
    
    # 3. 数据清洗和格式化
    print(f"处理医疗数据...")
    processed_data = []
    for item in raw_data:
        # 支持多种字段名
        query = item.get("query") or item.get("question") or item.get("input") or item.get("instruction", "")
        response = item.get("response") or item.get("answer") or item.get("output") or item.get("content", "")
        
        # 过滤太短的数据
        if len(query.strip()) > 5 and len(response.strip()) > 10:
            processed_data.append({
                "query": query.strip(),
                "response": response.strip()
            })
    
    print(f"   有效医疗数据: {len(processed_data)} 条")
    
    # 4. 计算自我认知数据重复次数
    if self_cognition_repeat is None:
        # 根据目标占比自动计算重复次数
        # target_ratio = self_cognition_count / total_count
        # self_cognition_count = len(self_cognition) * repeat
        # total_count = len(self_cognition) * repeat + len(processed_data)
        # target_ratio = (len(self_cognition) * repeat) / (len(self_cognition) * repeat + len(processed_data))
        # 解方程得到: repeat = (target_ratio * len(processed_data)) / (len(self_cognition) * (1 - target_ratio))
        if len(processed_data) > 0 and len(self_cognition) > 0:
            self_cognition_repeat = int((target_ratio * len(processed_data)) / (len(self_cognition) * (1 - target_ratio)))
            # 确保至少重复50次
            self_cognition_repeat = max(self_cognition_repeat, 50)
            print(f"   自动计算重复次数: {self_cognition_repeat} 次（目标占比: {target_ratio*100:.1f}%）")
        else:
            self_cognition_repeat = 100  # 默认值
            print(f"   使用默认重复次数: {self_cognition_repeat} 次")
    else:
        print(f"   使用指定重复次数: {self_cognition_repeat} 次")
    
    # 5. 合并自我认知数据（重复指定次数，提高权重）
    print(f"合并数据（自我认知数据重复{self_cognition_repeat}次以提高权重）...")
    # 将自我认知数据放在前面，并重复多次
    final_data = self_cognition * self_cognition_repeat + processed_data
    print(f"   合并后数据量: {len(final_data)} 条")
    actual_ratio = len(self_cognition) * self_cognition_repeat / len(final_data)
    print(f"   自我认知数据占比: {actual_ratio * 100:.2f}% ({len(self_cognition) * self_cognition_repeat} 条)")
    
    # 检查占比是否足够
    if actual_ratio < 0.05:
        print(f"   警告: 自我认知数据占比过低（{actual_ratio*100:.2f}%），建议至少达到5%以确保效果明显")
        print(f"   建议: 增加 --self_cognition_repeat 参数值，或使用 --target_ratio 参数（例如：--target_ratio 0.08）")
    elif actual_ratio >= 0.05:
        print(f"   良好: 自我认知数据占比达到 {actual_ratio*100:.2f}%，权重足够，效果应该明显")
    
    # 5. 打乱数据
    print(f"打乱数据顺序...")
    random.shuffle(final_data)
    
    # 6. 划分训练集和验证集（8:2）
    split_idx = int(len(final_data) * 0.8)
    train_data = final_data[:split_idx]
    val_data = final_data[split_idx:]
    
    # 7. 保存
    train_path = Path(output_dir) / "train.jsonl"
    val_path = Path(output_dir) / "val.jsonl"
    
    with open(train_path, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    with open(val_path, "w", encoding="utf-8") as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"\n数据处理完成！")
    print(f"   训练集：{len(train_data)} 条 -> {train_path}")
    print(f"   验证集：{len(val_data)} 条 -> {val_path}")
    print(f"   自我认知数据：{len(self_cognition)} 条（重复{self_cognition_repeat}次）")
    print(f"   医疗数据：{len(processed_data)} 条")
    
    return train_path, val_path

def check_data_quality(data_path):
    """检查数据质量"""
    data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    
    print(f"\n数据质量检查: {data_path}")
    print(f"   总数据量：{len(data)}")
    
    if len(data) == 0:
        print("   警告: 数据为空！")
        return
    
    avg_query_len = sum(len(d.get("query", "")) for d in data) / len(data)
    avg_response_len = sum(len(d.get("response", "")) for d in data) / len(data)
    
    print(f"   平均query长度：{avg_query_len:.1f} 字符")
    print(f"   平均response长度：{avg_response_len:.1f} 字符")
    
    # 检查空数据
    empty_query = sum(1 for d in data if not d.get("query", "").strip())
    empty_response = sum(1 for d in data if not d.get("response", "").strip())
    
    if empty_query > 0:
        print(f"   警告: 空query数量：{empty_query}")
    if empty_response > 0:
        print(f"   警告: 空response数量：{empty_response}")
    
    if empty_query == 0 and empty_response == 0:
        print(f"   数据质量良好")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="处理医疗数据，合并自我认知数据")
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="输入数据文件路径（支持JSON和JSONL格式）"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./processed_data",
        help="输出目录"
    )
    parser.add_argument(
        "--trainer_name",
        type=str,
        default="达星辰，陈欣悦，彭延浩",
        help="训练者姓名"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen3-1.7B",
        help="模型名称"
    )
    parser.add_argument(
        "--self_cognition_repeat",
        type=int,
        default=None,
        help="自我认知数据重复次数（如果未指定，则根据target_ratio自动计算，确保占比达到5-10%）"
    )
    parser.add_argument(
        "--target_ratio",
        type=float,
        default=0.08,
        help="目标自我认知数据占比（默认0.08，即8%）。如果self_cognition_repeat未指定，将自动计算重复次数以达到此占比"
    )
    parser.add_argument(
        "--self_cognition_file",
        type=str,
        default="self_cognition.jsonl",
        help="自我认知数据文件路径（如果存在则优先使用）"
    )
    parser.add_argument(
        "--check_quality",
        action="store_true",
        help="检查数据质量"
    )
    
    args = parser.parse_args()
    
    # 处理数据
    train_path, val_path = process_medical_data(
        args.input_file,
        args.output_dir,
        args.trainer_name,
        args.model_name,
        args.self_cognition_repeat,
        args.self_cognition_file,
        args.target_ratio
    )
    
    # 检查数据质量
    if args.check_quality:
        check_data_quality(train_path)
        check_data_quality(val_path)
