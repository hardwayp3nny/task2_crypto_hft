import os
import torch
import numpy as np
from erl_config import Config, build_env
from trade_simulator import EvalTradeSimulator
from erl_agent import AgentD3QN, AgentDoubleDQN, AgentTwinD3QN
from collections import Counter
from metrics import sharpe_ratio, max_drawdown, return_over_max_drawdown


def to_python_number(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().item()
    else:
        return x


class EnsembleEvaluator:
    def __init__(self, save_path, agent_classes, args: Config):
        # 设置随机种子
        x = 42
        torch.manual_seed(x)
        np.random.seed(x)
        torch.cuda.manual_seed_all(x)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
        self.save_path = save_path
        self.agent_classes = agent_classes

        # args
        self.args = args
        self.agents = []
        self.thresh = 0.001
        self.num_envs = 1
        self.state_dim = 8 + 2
        self.device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

        self.trade_env = build_env(args.env_class, args.env_args, gpu_id=args.gpu_id)

        self.current_btc = 0
        self.cash = [args.starting_cash]
        self.btc_assets = [0]
        # self.net_assets = [torch.tensor(args.starting_cash, device=self.device)]
        self.net_assets = [args.starting_cash]
        self.starting_cash = args.starting_cash

        # Ensure state dimensions match
        if self.state_dim != args.env_args["state_dim"]:
            print(f"Warning: Agent state_dim ({self.state_dim}) != Environment state_dim ({args.env_args['state_dim']})")
            self.state_dim = args.env_args["state_dim"]
        self.trade_log_path = os.path.join(save_path, "trade_log.csv")
        # 创建CSV文件并写入表头
        with open(self.trade_log_path, 'w', encoding='utf-8') as f:
            f.write("Step,Action,Price,Cash,BTC,BTC_Value,Total\n")
        self.signal_count = 0  # 连续信号计数
        self.last_signal = 0   # 上一个信号
        self.signal_window = []  # 存储移动窗口内的信号
        self.window_size = 79   # 移动窗口大小
        self.lock_signal = 0    # 锁定的信号
        self.lock_steps = 0     # 锁定剩余步数

    def load_agents(self):
        args = self.args
        for agent_class in self.agent_classes:
            agent = agent_class(
                args.net_dims,
                args.state_dim,
                args.action_dim,
                gpu_id=args.gpu_id,
                args=args,
            )
            agent_name = agent_class.__name__
            cwd = os.path.join(self.save_path, agent_name)
            print(f"Loading agent from: {os.path.abspath(cwd)}")
            # 检查模型文件是否存在
            model_path = os.path.join(cwd, "actor.pth")
            if os.path.exists(model_path):
                print(f"Found model file: {model_path}")
            else:
                print(f"Warning: Model file not found at {model_path}")
            agent.save_or_load_agent(cwd, if_save=False)  # Load agent
            self.agents.append(agent)

    def multi_trade(self):
        """Evaluation loop using ensemble of agents"""

        agents = self.agents
        trade_env = self.trade_env
        state = trade_env.reset()

        last_state = state
        last_price = 0

        positions = []
        action_ints = []
        correct_pred = []
        current_btcs = [self.current_btc]

        for _ in range(trade_env.max_step-1):
            actions = []
            intermediate_state = last_state

            # Collect actions from each agent
            for agent in agents:
                actor = agent.act
                tensor_state = torch.as_tensor(intermediate_state, dtype=torch.float32, device=agent.device)
                tensor_q_values = actor(tensor_state)
                tensor_action = tensor_q_values.argmax(dim=1)
                action = tensor_action.detach().cpu().unsqueeze(1)
                actions.append(action)

            # Debug agent decisions
            print(f"Individual agent actions: {[a.item() for a in actions]}")
            action = self._ensemble_action(actions=actions)
            action_int = action.item() - 1

            state, reward, done, _ = trade_env.step(action=action)

            action_ints.append(action_int)
            positions.append(trade_env.position)

            # Manually compute cumulative returns
            mid_price = trade_env.price_ary[trade_env.step_i, 2].to(self.device)
            mid_price_value = mid_price.item()

            new_cash = self.cash[-1]
            act1=0
            act2=0
            # 信号锁定逻辑
            if self.lock_steps > 0:
                action_int = self.lock_signal
                self.lock_steps -= 1
            else:
                # 信号锁定逻辑
                            original_action = action_int  # 保存原始信号
                            
                            # 更新移动窗口（无论是否处于锁定状态都继续更新）
                            self.signal_window.append(original_action)
                            if len(self.signal_window) > self.window_size:
                                self.signal_window.pop(0)
                            
                            # 计算窗口内的空头信号数量
                            short_signals = sum(1 for x in self.signal_window if x < 0)
                            
                            # 检查是否需要锁定或继续锁定
                            if short_signals >= 9:
                                self.lock_steps = 500
                                action_int = -1  # 强制执行空头
                                print(f"Locking/Maintaining short position due to {short_signals} short signals in window")
                            elif self.lock_steps > 0:
                                self.lock_steps -= 1
                                action_int = -1  # 维持空头

            # Modified trading logic with better price checks
            if action_int > 0:  # Buy signal
                act1 += 1
                if self.current_btc <0:
                    trade_value = 32*mid_price_value * (1 + self.args.env_args["slippage"])
                    new_cash -= trade_value
                    self.current_btc += 32
                    print(f"Executed BUY at {mid_price_value:.2f}")
                elif self.current_btc == 0:  # Can afford to buy
                    trade_value = 16*mid_price_value * (1 + self.args.env_args["slippage"])
                    new_cash -= trade_value
                    self.current_btc += 16
                    print(f"Executed BUY at {mid_price_value:.2f}")
            elif action_int < 0:  # Sell signal
                act2 += 1
                if self.current_btc > 0:
                    trade_value = 32*mid_price_value * (1 - self.args.env_args["slippage"])
                    new_cash += trade_value
                    self.current_btc -= 32
                    print(f"Executed SELL at {mid_price_value:.2f}")
                elif self.current_btc == 0:
                    trade_value = 16*mid_price_value * (1 - self.args.env_args["slippage"])
                    new_cash += trade_value
                    self.current_btc -= 16
                    print(f"Executed SELL at {mid_price_value:.2f}")


            self.cash.append(new_cash)
            btc_value = self.current_btc * mid_price.item()  # Calculate BTC value in cash
            self.btc_assets.append(btc_value)
            self.net_assets.append(new_cash + btc_value)  # Total assets = cash + BTC value

            # Print debug information for the first few trades
            if len(self.net_assets) > 0:
                btc_log = (f"Step {len(self.net_assets)}: Action={action_int}, "
                      f"Price={mid_price.item():.2f}, Cash={new_cash:.2f}, "
                      f"BTC={self.current_btc}, BTC Value={btc_value:.2f}, "
                      f"Total={self.net_assets[-1]:.2f}")
                print(btc_log)
                
                # 将交易信息写入CSV文件
                with open(self.trade_log_path, 'a', encoding='utf-8') as f:
                    f.write(f"{len(self.net_assets)},{action_int},{mid_price.item():.2f},"
                           f"{new_cash:.2f},{self.current_btc},{btc_value:.2f},{self.net_assets[-1]:.2f}\n")
            
            last_state = state

            # Log win rate
            if action_int == 1:
                correct_pred.append(1 if last_price < mid_price else -1 if last_price > mid_price else 0)
            elif action_int == -1:
                correct_pred.append(-1 if last_price < mid_price else 1 if last_price > mid_price else 0)
            else:
                correct_pred.append(0)

            last_price = mid_price
            current_btcs.append(self.current_btc)

        # Convert positions to CPU before saving if they're on GPU
        positions_np = [p.cpu().numpy() if torch.is_tensor(p) else p for p in positions]
        
        # Save results
        np.save(f"{self.save_path}_positions.npy", np.array(positions_np))
        np.save(f"{self.save_path}_net_assets.npy", np.array(self.net_assets))
        np.save(f"{self.save_path}_btc_positions.npy", np.array(self.btc_assets))
        np.save(f"{self.save_path}_correct_predictions.npy", np.array(correct_pred))

        # Compute metrics
        # 在计算指标之前添加调试信息
        print(f"Net assets history: {self.net_assets}")
        print(f"Returns statistics:")
        returns = np.diff(self.net_assets) / self.net_assets[:-1]
        print(f"- Mean return: {np.mean(returns)}")
        print(f"- Std return: {np.std(returns)}")
        print(f"- Min return: {np.min(returns)}")
        print(f"- Max return: {np.max(returns)}")
        print(f"- Number of trades: {len([x for x in action_ints if x != 0])}")
        print(f"-act1: {act1}，act2: {act2}")
        
        # 添加检查，避免除以零导致的无穷大
        if np.std(returns) == 0:
            print("Warning: Returns have zero standard deviation!")
            final_sharpe_ratio = 0
        else:
            final_sharpe_ratio = sharpe_ratio(returns)
            
        if len(returns) == 0:
            print("Warning: No trading returns generated!")
            final_max_drawdown = 0
            final_roma = 0
        else:
            final_max_drawdown = max_drawdown(returns)
            final_roma = return_over_max_drawdown(returns)
    
        print(f"Sharpe Ratio: {final_sharpe_ratio}")
        print(f"Max Drawdown: {final_max_drawdown}")
        print(f"Return over Max Drawdown: {final_roma}")

    def _ensemble_action(self, actions):
        """Returns the majority action among agents. Our code uses majority voting, you may change this to increase performance."""
        count = Counter([a.item() for a in actions])
        majority_action, _ = count.most_common(1)[0]
        return torch.tensor([[majority_action]], dtype=torch.int32)


def run_evaluation(save_path, agent_list):
    import sys
    print(f"\nInitializing evaluation...")
    
    # 检查数据配置
    from data_config import ConfigData
    config_data = ConfigData()
    print(f"\nData Configuration:")
    print(f"Factor array path: {os.path.abspath(config_data.predict_ary_path)}")
    print(f"Price data path: {os.path.abspath(config_data.csv_path)}")
    
    # 检查数据文件是否存在
    for path in [config_data.predict_ary_path, config_data.csv_path]:
        if os.path.exists(path):
            print(f"Found data file: {path}")
            print(f"File size: {os.path.getsize(path) / (1024*1024):.2f} MB")
        else:
            print(f"Warning: Data file not found at {path}")
    
    model_dir = os.path.abspath(save_path)
    print(f"\nLoading models from: {model_dir}")
    if os.path.exists(model_dir):
        print(f"Model directory contents:")
        for item in os.listdir(model_dir):
            item_path = os.path.join(model_dir, item)
            if os.path.isdir(item_path):
                print(f"  Directory: {item}")
                for subitem in os.listdir(item_path):
                    print(f"    - {subitem}")
            else:
                print(f"  File: {item}")
    else:
        print(f"Warning: Model directory not found at {model_dir}")
    
    print(f"Agent list: {[agent.__name__ for agent in agent_list]}")

    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else -1
    print(f"Using GPU ID: {gpu_id}")

    num_sims = 1
    num_ignore_step = 800
    max_position = 1
    step_gap = 16
    slippage = 7e-7

    max_step = (64000 - num_ignore_step) // step_gap

    env_args = {
        "env_name": "TradeSimulator-v0",
        "num_envs": num_sims,
        "max_step": max_step,
        "state_dim": 8 + 2,
        "action_dim": 3,
        "if_discrete": True,
        "max_position": max_position,
        "slippage": slippage,
        "num_sims": num_sims,
        "step_gap": step_gap,
        "dataset_path": config_data.csv_path,  # 使用配置中的路径
    }
    args = Config(agent_class=None, env_class=EvalTradeSimulator, env_args=env_args)
    args.gpu_id = gpu_id
    args.random_seed = gpu_id
    args.net_dims = (128, 128, 128)
    args.starting_cash = 1e6

    ensemble_evaluator = EnsembleEvaluator(
        save_path,
        agent_list,
        args,
    )
    print("\nLoading trained agents...")
    ensemble_evaluator.load_agents()
    print("Starting trading simulation...")
    ensemble_evaluator.multi_trade()


if __name__ == "__main__":
    save_path = "trained_agents"
    agent_list = [AgentD3QN, AgentDoubleDQN, AgentTwinD3QN]
    run_evaluation(save_path, agent_list)
    