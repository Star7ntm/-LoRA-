import matplotlib.pyplot as plt
import json
import os
import sys
from pathlib import Path

def find_latest_trainer_state(output_dir="./sft_output"):
    """查找最新的trainer_state.json文件"""
    output_path = Path(output_dir)
    
    # 首先检查根目录
    root_state = output_path / "trainer_state.json"
    if root_state.exists():
        return str(root_state)
    
    # 查找所有checkpoint目录
    checkpoints = sorted(output_path.glob("checkpoint-*"), key=lambda x: int(x.name.split("-")[-1]) if x.name.split("-")[-1].isdigit() else 0)
    
    if checkpoints:
        # 使用最新的checkpoint
        latest_checkpoint = checkpoints[-1]
        state_file = latest_checkpoint / "trainer_state.json"
        if state_file.exists():
            print(f"找到trainer_state.json: {state_file}")
            return str(state_file)
    
    return None

def plot_training_loss(trainer_state_path=None, output_dir="./loss_plots"):
    """从trainer_state.json中绘制Loss曲线并保存"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 如果没有指定路径，自动查找
    if trainer_state_path is None:
        # 从输出目录推断
        output_base = "./sft_output"
        trainer_state_path = find_latest_trainer_state(output_base)
        if trainer_state_path is None:
            print(f"错误：找不到 trainer_state.json 文件")
            print(f"   已搜索目录: {output_base}")
            print("   请确保训练已经完成，并且输出目录中有 trainer_state.json 文件")
            return
    elif not os.path.exists(trainer_state_path):
        # 如果指定的路径不存在，尝试自动查找
        print(f"警告: 指定的文件不存在: {trainer_state_path}")
        print("   尝试自动查找...")
        output_base = os.path.dirname(trainer_state_path) if os.path.dirname(trainer_state_path) else "./sft_output"
        trainer_state_path = find_latest_trainer_state(output_base)
        if trainer_state_path is None:
            print(f"错误：找不到 trainer_state.json 文件")
            return
    
    with open(trainer_state_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 提取loss数据
    steps = []
    train_losses = []
    eval_losses = []
    
    # 检查是否是loss_data.json格式（直接包含steps和train_losses）
    if "steps" in data and "train_losses" in data:
        # 直接使用loss_data.json格式
        steps = data["steps"]
        train_losses = data["train_losses"]
        if "eval_losses" in data and data["eval_losses"]:
            if isinstance(data["eval_losses"][0], (list, tuple)) and len(data["eval_losses"][0]) == 2:
                eval_losses = data["eval_losses"]
            else:
                # 如果是简单的列表，需要匹配steps
                eval_steps = data.get("eval_steps", steps)
                eval_losses = list(zip(eval_steps, data["eval_losses"]))
    else:
        # trainer_state.json格式（包含log_history）
        for log in data.get("log_history", []):
            if "loss" in log:
                steps.append(log.get("step", 0))
                train_losses.append(log["loss"])
            if "eval_loss" in log:
                eval_losses.append((log.get("step", 0), log["eval_loss"]))
    
    if not train_losses:
        print("错误: 没有找到loss数据")
        return
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 绘制图表
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 训练Loss
    axes[0].plot(steps, train_losses, 'b-', linewidth=2, label='训练Loss')
    axes[0].set_xlabel('训练步数')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('训练Loss曲线')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # 训练和验证Loss对比
    if eval_losses:
        eval_steps, eval_loss_values = zip(*eval_losses)
        axes[1].plot(steps, train_losses, 'b-', linewidth=2, label='训练Loss')
        axes[1].plot(eval_steps, eval_loss_values, 'r-', linewidth=2, label='验证Loss')
        axes[1].set_xlabel('训练步数')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('训练 vs 验证 Loss')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
    else:
        axes[1].plot(steps, train_losses, 'b-', linewidth=2, label='训练Loss')
        axes[1].set_xlabel('训练步数')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('训练Loss曲线（无验证数据）')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
    
    plt.tight_layout()
    
    # 保存图片
    plot_path = os.path.join(output_dir, "training_loss.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Loss曲线已保存到: {plot_path}")
    
    # 保存数据到JSON文件
    data = {
        "steps": steps,
        "train_losses": train_losses,
        "eval_losses": eval_losses if eval_losses else [],
        "final_train_loss": train_losses[-1] if train_losses else None,
        "final_eval_loss": eval_loss_values[-1] if eval_losses else None
    }
    json_path = os.path.join(output_dir, "loss_data.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Loss数据已保存到: {json_path}")
    
    # 打印统计信息
    print(f"\nLoss统计信息：")
    print(f"  总训练步数：{len(steps)}")
    print(f"  初始Loss：{train_losses[0]:.4f}")
    print(f"  最终Loss：{train_losses[-1]:.4f}")
    print(f"  Loss下降：{train_losses[0] - train_losses[-1]:.4f}")
    if eval_losses:
        print(f"  最终验证Loss：{eval_loss_values[-1]:.4f}")
        print(f"  训练-验证Loss差距：{abs(train_losses[-1] - eval_loss_values[-1]):.4f}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="绘制训练Loss曲线")
    parser.add_argument(
        "--trainer_state_path",
        type=str,
        default=None,
        help="trainer_state.json文件路径（如果不指定，会自动查找最新的checkpoint）"
    )
    parser.add_argument(
        "--output_base_dir",
        type=str,
        default="./sft_output",
        help="训练输出目录（用于自动查找trainer_state.json）"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./loss_plots",
        help="输出目录"
    )
    
    args = parser.parse_args()
    
    # 如果没有指定trainer_state_path，从output_base_dir查找
    if args.trainer_state_path is None:
        trainer_state_path = find_latest_trainer_state(args.output_base_dir)
        if trainer_state_path is None:
            print(f"错误：在 {args.output_base_dir} 中找不到 trainer_state.json")
            print("   请指定 --trainer_state_path 或确保训练已完成")
            sys.exit(1)
        args.trainer_state_path = trainer_state_path
    
    plot_training_loss(args.trainer_state_path, args.output_dir)

