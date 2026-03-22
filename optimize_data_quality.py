"""
数据质量优化脚本：筛选更少但更精确的高质量数据
目标：使用高质量数据获得更高的准确率和更低的Loss
"""
import json
import random
from pathlib import Path
from collections import Counter
import re

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
                for key in ["data", "items", "samples"]:
                    if key in data and isinstance(data[key], list):
                        return data[key]
                return [data]
        except:
            pass
    
    return []

def calculate_quality_score(item):
    """计算数据质量分数（0-100）"""
    query = item.get("query", "").strip()
    response = item.get("response", "").strip()
    
    score = 0
    
    # 1. 长度评分（30分）
    query_len = len(query)
    response_len = len(response)
    
    # Query长度：10-100字符为最佳（20分）
    if 10 <= query_len <= 100:
        score += 20
    elif 5 <= query_len < 10 or 100 < query_len <= 200:
        score += 10
    elif query_len > 200:
        score += 5
    
    # Response长度：50-500字符为最佳（10分）
    if 50 <= response_len <= 500:
        score += 10
    elif 20 <= response_len < 50 or 500 < response_len <= 1000:
        score += 5
    elif response_len > 1000:
        score += 2
    
    # 2. 内容质量评分（40分）
    # 检查是否包含有效的中医关键词
    tcm_keywords = ["中医", "中药", "针灸", "穴位", "经络", "气血", "阴阳", "五行", 
                    "辨证", "脉象", "舌苔", "方剂", "药材", "病症", "症状", "治疗",
                    "调理", "养生", "食疗", "按摩", "推拿", "艾灸"]
    
    keyword_count = sum(1 for keyword in tcm_keywords if keyword in (query + response))
    if keyword_count >= 3:
        score += 20
    elif keyword_count >= 2:
        score += 15
    elif keyword_count >= 1:
        score += 10
    
    # 检查响应是否包含具体建议（20分）
    if any(word in response for word in ["建议", "可以", "应该", "注意", "避免", "推荐"]):
        score += 20
    elif len(response) > 100:  # 长响应通常包含更多信息
        score += 10
    
    # 3. 结构质量评分（20分）
    # 检查是否有重复内容
    if query != response:  # 避免query和response相同
        score += 10
    
    # 检查是否有标点符号（说明格式良好）
    if any(punct in response for punct in ["。", "，", "、", "；", "："]):
        score += 10
    
    # 4. 信息密度评分（10分）
    # 检查是否包含数字（可能包含剂量、时间等信息）
    if re.search(r'\d+', response):
        score += 5
    
    # 检查是否包含专业术语
    if len(response.split()) > 10 or len(response) > 150:
        score += 5
    
    return min(score, 100)  # 限制在0-100

def filter_high_quality_data(data, min_score=60, max_samples=None):
    """筛选高质量数据"""
    print(f"\n开始数据质量筛选...")
    print(f"   原始数据量: {len(data)} 条")
    
    # 计算每条数据的质量分数
    scored_data = []
    for item in data:
        score = calculate_quality_score(item)
        scored_data.append((item, score))
    
    # 按分数排序（降序）
    scored_data.sort(key=lambda x: x[1], reverse=True)
    
    # 筛选高质量数据
    high_quality = [item for item, score in scored_data if score >= min_score]
    
    print(f"   质量分数 >= {min_score} 的数据: {len(high_quality)} 条")
    
    # 如果指定了最大样本数，取前N条
    if max_samples and len(high_quality) > max_samples:
        high_quality = high_quality[:max_samples]
        print(f"   限制到前 {max_samples} 条高质量数据")
    
    # 显示质量分布
    score_distribution = Counter(score for _, score in scored_data)
    print(f"\n质量分数分布:")
    for score_range in [(90, 100), (80, 89), (70, 79), (60, 69), (0, 59)]:
        count = sum(1 for _, score in scored_data if score_range[0] <= score <= score_range[1])
        print(f"   {score_range[0]}-{score_range[1]}分: {count} 条")
    
    return high_quality

def optimize_data_quality(input_file, output_dir="./processed_data_high_quality", 
                         self_cognition_file="self_cognition.jsonl",
                         target_ratio=0.15, min_quality_score=70, max_medical_samples=30000):
    """优化数据质量：筛选高质量数据并合并自我认知数据"""
    Path(output_dir).mkdir(exist_ok=True)
    
    print("="*60)
    print("数据质量优化（目标Loss < 0.3）")
    print("="*60)
    
    # 1. 加载自我认知数据（必须保留）
    self_cognition_path = Path(self_cognition_file)
    if self_cognition_path.exists():
        print(f"\n加载自我认知数据: {self_cognition_file}")
        self_cognition = load_data(self_cognition_file)
        print(f"   自我认知数据: {len(self_cognition)} 条")
    else:
        print(f"\n警告: 自我认知数据文件不存在: {self_cognition_file}")
        print("   将生成默认自我认知数据...")
        from generate_self_cognition import generate_self_cognition_data
        self_cognition = generate_self_cognition_data()
        print(f"   生成自我认知数据: {len(self_cognition)} 条")
    
    # 2. 加载原始医疗数据
    print(f"\n加载原始医疗数据: {input_file}")
    raw_data = load_data(input_file)
    print(f"   原始数据量: {len(raw_data)} 条")
    
    # 3. 数据清洗和格式化
    print(f"\n数据清洗和格式化...")
    processed_data = []
    for item in raw_data:
        query = item.get("query") or item.get("question") or item.get("input") or item.get("instruction", "")
        response = item.get("response") or item.get("answer") or item.get("output") or item.get("content", "")
        
        # 基本过滤
        if len(query.strip()) > 5 and len(response.strip()) > 10:
            processed_data.append({
                "query": query.strip(),
                "response": response.strip()
            })
    
    print(f"   有效医疗数据: {len(processed_data)} 条")
    
    # 4. 筛选高质量数据
    high_quality_medical = filter_high_quality_data(
        processed_data, 
        min_score=min_quality_score,
        max_samples=max_medical_samples
    )
    
    print(f"\n高质量医疗数据: {len(high_quality_medical)} 条")
    print(f"   数据减少: {len(processed_data)} -> {len(high_quality_medical)} ({len(high_quality_medical)/len(processed_data)*100:.1f}%)")
    
    # 5. 计算自我认知数据重复次数
    if len(high_quality_medical) > 0 and len(self_cognition) > 0:
        self_cognition_repeat = int((target_ratio * len(high_quality_medical)) / (len(self_cognition) * (1 - target_ratio)))
        self_cognition_repeat = max(self_cognition_repeat, 50)  # 至少重复50次
        print(f"\n自我认知数据重复次数: {self_cognition_repeat} 次（目标占比: {target_ratio*100:.1f}%）")
    else:
        self_cognition_repeat = 100
        print(f"\n使用默认重复次数: {self_cognition_repeat} 次")
    
    # 6. 合并数据
    print(f"\n合并数据...")
    final_data = self_cognition * self_cognition_repeat + high_quality_medical
    print(f"   合并后数据量: {len(final_data)} 条")
    
    actual_ratio = len(self_cognition) * self_cognition_repeat / len(final_data)
    print(f"   自我认知数据占比: {actual_ratio * 100:.2f}% ({len(self_cognition) * self_cognition_repeat} 条)")
    print(f"   高质量医疗数据: {len(high_quality_medical)} 条")
    
    # 7. 打乱数据
    print(f"\n打乱数据顺序...")
    random.shuffle(final_data)
    
    # 8. 划分训练集和验证集（8:2）
    split_idx = int(len(final_data) * 0.8)
    train_data = final_data[:split_idx]
    val_data = final_data[split_idx:]
    
    # 9. 保存
    train_path = Path(output_dir) / "train.jsonl"
    val_path = Path(output_dir) / "val.jsonl"
    
    with open(train_path, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    with open(val_path, "w", encoding="utf-8") as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"\n" + "="*60)
    print("数据优化完成！")
    print("="*60)
    print(f"   训练集：{len(train_data)} 条 -> {train_path}")
    print(f"   验证集：{len(val_data)} 条 -> {val_path}")
    print(f"   自我认知数据：{len(self_cognition)} 条（重复{self_cognition_repeat}次）")
    print(f"   高质量医疗数据：{len(high_quality_medical)} 条")
    print(f"   数据减少比例：{len(processed_data)} -> {len(high_quality_medical)} ({len(high_quality_medical)/len(processed_data)*100:.1f}%)")
    print(f"   自我认知数据占比：{actual_ratio * 100:.2f}%")
    
    return train_path, val_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="优化数据质量：筛选高质量数据")
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="输入数据文件路径"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./processed_data_high_quality",
        help="输出目录"
    )
    parser.add_argument(
        "--self_cognition_file",
        type=str,
        default="self_cognition.jsonl",
        help="自我认知数据文件路径"
    )
    parser.add_argument(
        "--target_ratio",
        type=float,
        default=0.15,
        help="目标自我认知数据占比（默认0.15，即15%）"
    )
    parser.add_argument(
        "--min_quality_score",
        type=int,
        default=70,
        help="最低质量分数（默认70分，范围0-100）"
    )
    parser.add_argument(
        "--max_medical_samples",
        type=int,
        default=30000,
        help="最大医疗数据样本数（默认30000条）"
    )
    
    args = parser.parse_args()
    
    optimize_data_quality(
        args.input_file,
        args.output_dir,
        args.self_cognition_file,
        args.target_ratio,
        args.min_quality_score,
        args.max_medical_samples
    )

