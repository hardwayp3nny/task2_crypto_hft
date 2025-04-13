import os
import torch
import numpy as np
from collections import defaultdict
from erl_config import Config, build_env
from erl_agent import AgentD3QN
from trade_simulator import TradeSimulator, EvalTradeSimulator
from erl_net import QNetTwinDuel
from torch.serialization import safe_globals, add_safe_globals
from torch.nn import Sequential, Linear, ReLU, Softmax
from torch.optim import AdamW

def validate_latest_model(model_dir="TradeSimulator-v0_D3QN_0", gpu_id=-1):
    """验证最新的模型并生成position文件"""
    
    # 添加所有需要的安全加载类
    add_safe_globals([
        QNetTwinDuel,
        Sequential,
        Linear,
        ReLU,
        AdamW,
        Softmax,
        defaultdict,
        dict,  # 添加这行
    ])
    
    # 环境参数设置
    num_sims = 512
    num_ignore_step = 60
    max_position = 1
    step_gap = 2
    slippage = 7e-7
    max_step = (4800 - num_ignore_step) // step_gap

    env_args = {
        "env_name": "TradeSimulator-v0",
        "num_envs": num_sims,
        "max_step": max_step,
        "state_dim": 8 + 2,  # factor_dim + (position, holding)
        "action_dim": 3,  # long, 0, short
        "if_discrete": True,
        "max_position": max_position,
        "slippage": slippage,
        "num_sims": num_sims,
        "step_gap": step_gap,
    }

    # 初始化环境和智能体
    args = Config(agent_class=AgentD3QN, env_class=TradeSimulator, env_args=env_args)
    args.gpu_id = gpu_id
    args.eval_env_class = EvalTradeSimulator
    args.eval_env_args = env_args.copy()
    
    sim = build_env(args.eval_env_class, args.eval_env_args, gpu_id=args.gpu_id)
    agent = args.agent_class(args.net_dims, args.state_dim, args.action_dim, gpu_id=args.gpu_id)

    # 修改模型加载部分
    agent.save_or_load_agent(cwd=model_dir, if_save=False)
    actor_files = [f for f in os.listdir(model_dir) if f.startswith('actor_') and f.endswith('.pth')]
    latest_actor = sorted(actor_files, key=lambda x: int(x.split('_')[1].split('.')[0]))[-1]
    print(f"使用模型: {latest_actor}")
    
    # 修改加载方式
    agent.act.load_state_dict(
        torch.load(
            f"{model_dir}/{latest_actor}", 
            map_location=agent.device,
            weights_only=False  # 添加这个参数
        ).state_dict()
    )

    # 开始验证
    actor = agent.act
    device = agent.device
    thresh = 0.001
    state = sim.reset()

    position_ary = []
    trade_ary = []
    q_values_ary = []
    
    print("开始验证...")
    for i in range(sim.max_step):
        tensor_state = torch.as_tensor(state, dtype=torch.float32, device=device)
        tensor_q_values = actor(tensor_state)
        tensor_action = tensor_q_values.argmax(dim=1)

        mask_zero_position = sim.position.eq(0)
        mask_q_values = (
            tensor_q_values.max(dim=1)[0] - tensor_q_values.mean(dim=1)
        ).lt(torch.where(tensor_action.eq(2), thresh, thresh))
        mask = torch.logical_and(mask_zero_position, mask_q_values)
        tensor_action[mask] = 1

        action = tensor_action.detach().cpu().unsqueeze(1)
        state, reward, done, info_dict = sim.step(action=action)

        trade_ary.append(sim.action_int.data.cpu().numpy())
        position_ary.append(sim.position.data.cpu().numpy())
        q_values_ary.append(tensor_q_values.data.cpu().numpy())

    # 保存结果
    save_path = "erl_run_valid_position.npy"
    position_ary = np.stack(position_ary, axis=0)
    np.save(save_path, position_ary)
    print(f"验证完成，结果已保存到: {save_path}")

if __name__ == "__main__":
    validate_latest_model()