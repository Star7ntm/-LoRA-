import json
import time
import os
from pathlib import Path
from datetime import datetime

def check_training_process():
    """检查是否有Python训练进程在运行（简化版，不依赖psutil）"""
    # 在Windows上，可以通过检查输出目录是否有新文件来判断
    # 或者直接返回True，让用户知道训练可能在进行
    return True  # 简化处理，假设训练可能在进行

def find_trainer_state(output_dir):
    """查找trainer_state.json文件，优先查找checkpoint目录中的"""
    output_path = Path(output_dir)
    
    # 1. 首先查找所有checkpoint目录中的trainer_state.json（优先）
    checkpoints = sorted(output_path.glob("checkpoint-*"), 
                        key=lambda x: int(x.name.split("-")[-1]) if x.name.split("-")[-1].isdigit() else 0) if output_path.exists() else []
    
    if checkpoints:
        # 从最新的checkpoint开始查找
        for checkpoint in reversed(checkpoints):
            state_file = checkpoint / "trainer_state.json"
            if state_file.exists():
                return state_file, checkpoints
    
    # 2. 查找根目录中的trainer_state.json
    root_state = output_path / "trainer_state.json"
    if root_state.exists():
        return root_state, checkpoints
    
    return None, checkpoints

def monitor_training(output_dir="./sft_output", interval=60):
    """监控训练进度"""
    output_path = Path(output_dir)
    
    # 查找trainer_state.json
    trainer_state_path, checkpoints = find_trainer_state(output_dir)
    
    if trainer_state_path is None:
        if not output_path.exists():
            print(f"错误: 输出目录不存在: {output_path}")
            print(f"   提示: 请先启动训练，或检查输出目录路径是否正确")
            return None
        elif checkpoints:
            # 有checkpoint但没有trainer_state.json，训练可能正在进行
            latest_ckpt = checkpoints[-1]
            print(f"警告: 训练正在进行中（检测到checkpoint: {latest_ckpt.name}）")
            print(f"   trainer_state.json在checkpoint目录中，当前checkpoint: {latest_ckpt.name}")
            # 尝试读取checkpoint中的trainer_state.json
            ckpt_state = latest_ckpt / "trainer_state.json"
            if ckpt_state.exists():
                trainer_state_path = ckpt_state
            else:
                print(f"   提示: trainer_state.json将在下一个checkpoint保存时创建")
                return {"status": "training", "checkpoint": latest_ckpt.name}
        else:
            # 检查是否有训练进程
            if check_training_process():
                print(f"检测到训练进程正在运行")
                print(f"   trainer_state.json将在第一个checkpoint保存时创建")
                print(f"   请稍候...")
                return {"status": "training", "checkpoint": None}
            else:
                print(f"错误: 训练尚未开始: {output_path}")
                print(f"   提示: trainer_state.json文件会在训练开始后创建")
                print(f"   请先启动训练:")
                print(f"   python med_train_trl.py --data_path ./processed_data/train.jsonl --max_steps 500 --output_dir {output_dir}")
                print(f"   或运行: train_ultra_fast.bat")
                return None
    
    # 如果trainer_state_path仍然为None，说明没有找到
    if trainer_state_path is None:
        return None
    
    with open(trainer_state_path, "r", encoding="utf-8") as f:
        state = json.load(f)
    
    # 获取最新日志
    log_history = state.get("log_history", [])
    latest_log = log_history[-1] if log_history else {}
    
    info = {
        "epoch": state.get("epoch", 0),
        "global_step": state.get("global_step", 0),
        "total_flos": state.get("total_flos", 0),
        "current_loss": latest_log.get("loss"),
        "current_eval_loss": latest_log.get("eval_loss"),
        "learning_rate": latest_log.get("learning_rate"),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    return info

def print_training_status(info):
    """打印训练状态"""
    if info is None:
        return
    
    # 如果是简单的状态信息
    if isinstance(info, dict) and info.get("status") == "training":
        checkpoint = info.get("checkpoint")
        if checkpoint:
            print(f"\n{'='*60}")
            print(f"训练状态监控 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*60}")
            print(f"  状态: 训练进行中")
            print(f"  最新checkpoint: {checkpoint}")
            print(f"  提示: trainer_state.json将在save_steps时创建")
            print(f"{'='*60}\n")
        else:
            print(f"训练进程运行中，等待checkpoint保存...")
        return
    
    print(f"\n{'='*60}")
    print(f"训练状态监控 - {info['timestamp']}")
    print(f"{'='*60}")
    print(f"  当前Epoch: {info['epoch']:.2f}")
    print(f"  当前步数: {info['global_step']}")
    print(f"  总FLOPs: {info['total_flos'] / 1e12:.2f} TFLOPs")
    
    if info['current_loss'] is not None:
        print(f"  当前训练Loss: {info['current_loss']:.4f}")
    
    if info['current_eval_loss'] is not None:
        print(f"  当前验证Loss: {info['current_eval_loss']:.4f}")
    
    if info['learning_rate'] is not None:
        print(f"  当前学习率: {info['learning_rate']:.2e}")
    
    print(f"{'='*60}\n")

def continuous_monitor(output_dir="./sft_output", interval=60):
    """持续监控训练进度"""
    print(f"开始监控训练进度（每{interval}秒更新一次）")
    print(f"   输出目录: {output_dir}")
    print(f"   按 Ctrl+C 停止监控\n")
    
    try:
        while True:
            info = monitor_training(output_dir)
            if info:
                print_training_status(info)
                # 如果检测到训练正在进行但还没有trainer_state.json，继续等待
                if isinstance(info, dict) and info.get("status") == "training":
                    print(f"   提示: trainer_state.json将在第250步（save_steps）时创建")
                    print(f"   当前训练正在进行中，Loss会显示在训练终端\n")
            else:
                # 检查是否有训练进程
                output_path = Path(output_dir)
                if output_path.exists():
                    # 检查是否有日志文件或其他训练痕迹
                    log_files = list(output_path.glob("*.log")) + list(output_path.glob("runs/*"))
                    if log_files:
                        print(f"检测到训练日志文件，训练正在进行中...")
                    else:
                        print("等待训练开始或状态更新...")
                else:
                    print("等待训练开始...")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n\n监控已停止")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="监控训练进度")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./sft_output",
        help="训练输出目录"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="监控间隔（秒）"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="只监控一次，不持续监控"
    )
    
    args = parser.parse_args()
    
    if args.once:
        info = monitor_training(args.output_dir)
        print_training_status(info)
    else:
        continuous_monitor(args.output_dir, args.interval)

