
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
import numpy as np
def transform_matrix_softmax(alpha_dict):
    node_num = 11
    adj = torch.zeros((node_num, node_num))

    # 遍历cell节点
    for i in range(len(alpha_dict)):
        #[O4,O5],[O6,O7],[O8,O9]
        # 对每个连接选择使用Gumbel-Softmax进行采样，得到的是概率分布
        connect_probs = F.softmax(alpha_dict[f'log_alpha_{i}'], dim = 0)
        # O4
        for j in node_predecessors[i][0:-1]:  # 遍历可能的连接点
            if j == 0 or j == 1:
                adj[j][2*i + 4] = connect_probs[j]  # 特殊节点到当前cell节点的连接
            else:
                # 对于其他连接，我们简化逻辑，直接使用概率值
                # 注意：这里的实现需要根据实际场景调整
                # 可能的概率都要加上
                # adj[j, i + 2] = connect_probs[j % len(connect_probs)]
                adj[(j - 2) * 2 + 2][2*i + 4] = connect_probs[j]
                adj[(j - 2) * 2 + 3][2*i + 4] = connect_probs[j]
        #O5
        for k in node_predecessors[i][1:]:  # 遍历可能的连接点
            if k == 0 or k == 1:
                adj[k][ 2*i + 5] = connect_probs[k]  # 特殊节点到当前cell节点的连接
            else:
                # 对于其他连接，我们简化逻辑，直接使用概率值
                # 注意：这里的实现需要根据实际场景调整
                # 可能的概率都要加上
                # adj[j, i + 2] = connect_probs[j % len(connect_probs)]
                adj[(k - 2) * 2 + 2][2*i + 5] = connect_probs[k]
                adj[(k - 2) * 2 + 3][2*i + 5] = connect_probs[k]


    adj[2:-1, -1] = 1  # 假设每个cell节点直接连接到输出节点
    adj[0, 2] = 1
    adj[1, 3] = 1

    return adj


node_predecessors = [
    [0, 1, 2],                  # 节点3
    [0, 1, 2, 3],               # 节点4
    [0, 1, 2, 3, 4],            # 节点5
]

log_edge_index_alpha_dict = torch.nn.ParameterDict()

# 遍历所有节点，只为node_pred_train_flag为True的节点创建log_alpha
for node_idx, preds in enumerate(node_predecessors):
    # 创建一个Parameter对象，维度是该节点前序节点的数量
    log_alpha = Parameter(torch.zeros(len(preds)).normal_(1, 0.01), requires_grad=True)
    log_edge_index_alpha_dict[f'log_alpha_{node_idx}'] = log_alpha

# 设置优化器
optimizer = optim.Adam(log_edge_index_alpha_dict.parameters(), lr=0.01)

# 简单的训练循环
for step in range(100):
    optimizer.zero_grad()
    adj = transform_matrix_gumbel(log_edge_index_alpha_dict)
    # 假设损失函数是adj矩阵所有元素的平方和
    loss = adj.pow(2).sum()
    loss.backward()
    optimizer.step()

    if step % 10 == 0:
        print(f"Step {step}, Loss: {loss.item()}")
        print(adj)
