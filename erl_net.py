import torch
import torch.nn as nn

TEN = torch.Tensor


class QNetBase(nn.Module):  # nn.Module is a standard PyTorch Network
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.explore_rate = 0.125
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net = None  # build_mlp(dims=[state_dim + action_dim, *dims, 1])

        self.state_avg = nn.Parameter(torch.zeros((state_dim,)), requires_grad=False)
        self.state_std = nn.Parameter(torch.ones((state_dim,)), requires_grad=False)
        self.value_avg = nn.Parameter(torch.zeros((1,)), requires_grad=False)
        self.value_std = nn.Parameter(torch.ones((1,)), requires_grad=False)

    def state_norm(self, state: TEN) -> TEN:
        return (state - self.state_avg) / self.state_std

    def value_re_norm(self, value: TEN) -> TEN:
        return value * self.value_std + self.value_avg


class QNetTwin(QNetBase):  # Double DQN
    def __init__(self, dims: [int], state_dim: int, action_dim: int):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        self.net_state = build_mlp(dims=[state_dim, *dims])
        self.net_val1 = build_mlp(dims=[dims[-1], action_dim])  # Q value 1
        self.net_val2 = build_mlp(dims=[dims[-1], action_dim])  # Q value 2
        self.soft_max = nn.Softmax(dim=1)

        layer_init_with_orthogonal(self.net_val1[-1], std=0.1)
        layer_init_with_orthogonal(self.net_val2[-1], std=0.1)

    def forward(self, state):
        state = self.state_norm(state)
        s_enc = self.net_state(state)  # encoded state
        q_val = self.net_val1(s_enc)  # q value
        return q_val  # one group of Q values

    def get_q1_q2(self, state):
        state = self.state_norm(state)
        s_enc = self.net_state(state)  # encoded state
        q_val1 = self.net_val1(s_enc)  # q value 1
        q_val1 = self.value_re_norm(q_val1)
        q_val2 = self.net_val2(s_enc)  # q value 2
        q_val2 = self.value_re_norm(q_val2)
        return q_val1, q_val2  # two groups of Q values

    def get_action(self, state):
        state = self.state_norm(state)
        s_enc = self.net_state(state)  # encoded state
        q_val = self.net_val1(s_enc)  # q value
        if self.explore_rate < torch.rand(1):
            action = q_val.argmax(dim=1, keepdim=True)
        else:
            # a_prob = self.soft_max(q_val)
            # action = torch.multinomial(a_prob, num_samples=1)
            action = torch.randint(self.action_dim, size=(state.shape[0], 1))
        return action


class QNetTwinDuel(QNetBase):  # D3QN: Dueling Double DQN
    def __init__(self, dims: [int], state_dim: int, action_dim: int):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        self.net_state = build_mlp(dims=[state_dim, *dims])
        self.net_adv1 = build_mlp(dims=[dims[-1], 1])  # advantage value 1
        self.net_val1 = build_mlp(dims=[dims[-1], action_dim])  # Q value 1
        self.net_adv2 = build_mlp(dims=[dims[-1], 1])  # advantage value 2
        self.net_val2 = build_mlp(dims=[dims[-1], action_dim])  # Q value 2
        self.soft_max = nn.Softmax(dim=1)

        layer_init_with_orthogonal(self.net_adv1[-1], std=0.1)
        layer_init_with_orthogonal(self.net_val1[-1], std=0.1)
        layer_init_with_orthogonal(self.net_adv2[-1], std=0.1)
        layer_init_with_orthogonal(self.net_val2[-1], std=0.1)

        # Initialize trading-related attributes
        self.trade_counts = torch.zeros(action_dim, device=self.net_state[0].weight.device)
        self.last_actions = None
        self.last_action_time = None
        self.min_hold_steps = 5  # Minimum holding period
        self.trade_balance_penalty = 0.1  # Penalty coefficient for buy-sell imbalance

    def forward(self, state):
        state = self.state_norm(state)
        s_enc = self.net_state(state)  # encoded state
        q_val = self.net_val1(s_enc)  # q value
        q_adv = self.net_adv1(s_enc)  # advantage value
        value = q_val - q_val.mean(dim=1, keepdim=True) + q_adv  # one dueling Q value
        value = self.value_re_norm(value)
        return value

    def get_action(self, state):
        state = self.state_norm(state)
        s_enc = self.net_state(state)
        q_val = self.net_val1(s_enc)
        
        # Calculate buy-sell imbalance penalty
        trade_imbalance = torch.abs(self.trade_counts[0] - self.trade_counts[2]) / (self.trade_counts.sum() + 1)
        balance_penalty = trade_imbalance * self.trade_balance_penalty
        
        # Apply penalty to Q values
        value = q_val - q_val.mean(dim=1, keepdim=True) + q_adv
        value = value - balance_penalty
        
        value = self.value_re_norm(value)
        return value

    def get_action(self, state):
        state = self.state_norm(state)
        s_enc = self.net_state(state)
        q_val = self.net_val1(s_enc)
        
        batch_size = state.shape[0]
        current_step = torch.ones(batch_size, 1, device=state.device)
        
        if self.last_actions is None:
            self.last_actions = torch.ones(batch_size, 1, device=state.device)
            self.last_action_time = torch.zeros(batch_size, 1, device=state.device)
        
        # Calculate holding time
        hold_time = current_step - self.last_action_time
        
        # Create action mask to prevent rapid position flipping
        mask = torch.ones_like(q_val)
        for i in range(batch_size):
            if hold_time[i] < self.min_hold_steps:
                if self.last_actions[i] == 0:
                    mask[i, 2] = float('-1e9')  
                elif self.last_actions[i] == 2:
                    mask[i, 0] = float('-1e9') 
        
        q_val = q_val + mask
        
        if self.explore_rate < torch.rand(1):
            action = q_val.argmax(dim=1, keepdim=True)
        else:
            action = torch.randint(self.action_dim, size=(state.shape[0], 1), device=state.device)
        
        # Update trade statistics and history
        for a in action.to(self.trade_counts.device):  # Ensure device matching
            self.trade_counts[a] += 1
        self.last_actions = action
        self.last_action_time = current_step
        
        return action


def build_mlp(dims: [int], activation: nn = None, if_raw_out: bool = True) -> nn.Sequential:
    """
    build MLP (MultiLayer Perceptron)

    dims: the middle dimension, `dims[-1]` is the output dimension of this network
    activation: the activation function
    if_remove_out_layer: if remove the activation function of the output layer.
    """
    if activation is None:
        activation = nn.ReLU
    net_list = []
    for i in range(len(dims) - 1):
        net_list.extend([nn.Linear(dims[i], dims[i + 1]), activation()])
    if if_raw_out:
        del net_list[-1]  # delete the activation function of the output layer to keep raw output
    return nn.Sequential(*net_list)


def layer_init_with_orthogonal(layer, std=1.0, bias_const=1e-6):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)