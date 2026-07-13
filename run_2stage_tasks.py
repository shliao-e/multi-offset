import subprocess
import os
import time
import sys

# ================= 配置区域 =================
# train2 阶段想要跑的 lmbda 列表（对应两个不同的码率点任务）
lmbda_list = [4] 

# 基础参数配置
cuda_ids = [0]          # 可用的显卡 ID
use_ckpt = 0               # 是否使用 checkpoint
train_batch_size = 8       # 你的 batch_size

train1 = "train_script_rgb/pretrain.py"
train2 = "train_script_rgb/finetune.py"

# 日志存放目录
log_dir = "train_logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
# ===========================================


# ========================================================
# 🚀 阶段一：顺序执行 train1 (单一任务，无 lambda 区分)
# ========================================================
print("==================================================")
print(" 🟢 正在启动 阶段一：train1 (Pretrain) 单一任务")
print("==================================================")

# 默认将 train1 跑在第一个可用的 GPU 上
train1_gpu = cuda_ids[0]

cmd_train1 = [
    "python", train1,
    "--cuda_id", str(train1_gpu),
    "--use_ckpt", str(use_ckpt),
    "--train_batch_size", str(train_batch_size)
    # 注：此处已去掉 --lmbda 参数
]

log_file_train1 = open(f"{log_dir}/train1_pretrain_gpu_{train1_gpu}.log", "w")
print(f"👉 train1 已启动 (GPU {train1_gpu})，正在运行，请稍候...")

p_train1 = subprocess.Popen(cmd_train1, stdout=log_file_train1, stderr=subprocess.STDOUT)

try:
    # 关键点：强行等待 train1 结束
    p_train1.wait()
    log_file_train1.close()
    
    if p_train1.returncode != 0:
        print(f"❌ 错误：train1 异常退出 (退出码: {p_train1.returncode})，将不会执行 train2！")
        sys.exit(1)
        
    print("✅ 成功：train1 阶段已全部完成！")

except KeyboardInterrupt:
    print("\n⚠️ 收到中断信号，正在终止 train1 任务...")
    p_train1.terminate()
    log_file_train1.close()
    print("🛑 train1 已停止，脚本退出。")
    sys.exit(1)


# 两个阶段之间稍微停顿 5 秒，预留显存释放和 IO 缓冲时间
print("\n系统将在 5 秒后自动进入下一阶段...")
time.sleep(5)


# ========================================================
# 🚀 阶段二：顺序触发 train2 (区分码率点，并行执行两个任务)
# ========================================================
print("==================================================")
print(" 🔵 正在启动 阶段二：train2 (Finetune) 多码率任务")
print("==================================================")

processes = []
log_files = []

for i, lmbda in enumerate(lmbda_list):
    # 轮询分配 GPU: 任务0 -> GPU0, 任务1 -> GPU1
    gpu_id = cuda_ids[i % len(cuda_ids)]
    
    # 构建 train2 运行命令 (带上对应的 --lmbda)
    cmd_train2 = [
        "python", train2,
        "--cuda_id", str(gpu_id),
        "--lmbda", str(lmbda),
        "--use_ckpt", str(use_ckpt),
        "--train_batch_size", str(train_batch_size)
    ]
    
    log_file_train2 = open(f"{log_dir}/train2_lmbda_{lmbda}_gpu_{gpu_id}.log", "w")
    log_files.append(log_file_train2)
    
    print(f"👉 启动 train2 任务：lmbda={lmbda} | 分配至 GPU {gpu_id}")
    
    # 异步启动进程
    p_train2 = subprocess.Popen(cmd_train2, stdout=log_file_train2, stderr=subprocess.STDOUT)
    processes.append(p_train2)
    
    # 防止多进程同时加载模型造成 IO 阻塞或显存瞬间激增
    time.sleep(5)

print(f"\n✅ 阶段二的 {len(lmbda_list)} 个任务已全部部署完毕！")
print(f"📝 运行日志保存在 {log_dir}/ 目录下。")

# 等待所有 train2 进程结束
try:
    for p in processes:
        p.wait()
    print("\n🎉【全部完成】train1 和 train2 所有任务已顺利结束！")
except KeyboardInterrupt:
    print("\n⚠️ 收到中断信号，正在终止所有 train2 运行中的训练任务...")
    for p in processes:
        p.terminate()
    print("🛑 所有 train2 任务已停止。")
finally:
    # 确保关闭所有打开的日志文件句柄
    for f in log_files:
        f.close()