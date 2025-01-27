import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("/home/wanglei/AAAI/lean_ATG/leanproject/leandojo-lean4-tacgen-byt5-small")       # Or "lean3" -> "lean4"
model = AutoModelForSeq2SeqLM.from_pretrained("/home/wanglei/AAAI/lean_ATG/leanproject/leandojo-lean4-tacgen-byt5-small")   # Or "lean3" -> "lean4"

class policy_model(nn.Module):

    def __init__(self, feature_size, device):
        super(policy_model, self).__init__()

        self.device = device
        self.feature_size = feature_size

        self.action1 = nn.Linear(in_features=feature_size, out_features=64)
        self.action2 = nn.Linear(in_features=64, out_features=64)

        self.action_head = nn.Linear(in_features=64, out_features=1)

    def forward(self, x):
        # print("hello")
        # print(x)

        x = F.relu(self.action1(x))
        x = F.relu(self.action2(x))

        action_logits = self.action_head(x)

        return torch.sigmoid(action_logits)

    def predict(self, state_policy):
        # state_policy = torch.FloatTensor(state_policy.astype(np.float32)).to(self.device) # 将state_policy从NumPy数组转换为PyTorch张量，并将其数据类型设置为32位浮点数
        # state_policy = state_policy.view(1, self.size) #对state_policy进行形状变换，将其视图更改为1行self.size列的矩阵

        state_policy = torch.FloatTensor(state_policy.astype(np.float32)).to(self.device)
        # print("hellohello")
        # print(state_policy)

        state_policy = state_policy.view(1, self.feature_size)
        self.eval()  # 将模型切换到评估模式。在评估模式下，模型不会更新梯度，这在推断阶段是有用的。

        with torch.no_grad():
            pi = self.forward(state_policy)

        return pi.data.cpu().numpy()[0]


class value_model(nn.Module):

    def __init__(self, feature_size, device):
        super(value_model, self).__init__()

        self.device = device
        self.feature_size = feature_size

        self.value1 = nn.Linear(in_features=feature_size, out_features=64)
        self.value2 = nn.Linear(in_features=64, out_features=64)

        self.value_head = nn.Linear(in_features=64, out_features=1)

    def forward(self, x):
        x = F.relu(self.value1(x))
        x = F.relu(self.value2(x))
        value_logit = self.value_head(x)

        return torch.tanh(value_logit)

    def predict(self, state):
        # state = torch.FloatTensor(state.astype(np.float32)).to(self.device)
        # state = state.view(1, self.size)

        state = torch.FloatTensor(state.astype(np.float32)).to(self.device)
        state = state.view(1, self.feature_size)
        self.eval()

        with torch.no_grad():
            v = self.forward(state)

        return v.data.cpu().numpy()[0]


