import os
import torch
import torch.nn as nn

def run_training():
    import sys
    sys.path.append("f:/finrl/FinRL_Contest_2025/Task_2_FinRL_AlphaSeek_Crypto")
    from erl_run import run
    
    # 检查GPU可用性
    if not torch.cuda.is_available():
        print("CUDA不可用，请检查GPU设置")
        exit(1)
    
    # 获取可用的GPU设备
    device_ids = list(range(torch.cuda.device_count()))
    world_size = len(device_ids)  # 获取GPU数量作为world_size
    print(f"检测到 {world_size} 个GPU设备")
    print(f"使用的GPU设备: {device_ids}")
    
    # 运行训练，只传递gpu_ids参数
    run(gpu_ids=device_ids)

if __name__ == "__main__":
    run_training()