
import torch
import torch.nn.functional as F
import torch.optim as optim

def gumbel_softmax(logits, temperature=1.0, dim=-1):
    gumbels = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)  # Gumbel噪声
    y = logits + gumbels
    return F.softmax(y / temperature, dim=dim)

def transform_matrix_gumbel(alpha, cell_size, temperature=0.1):
    node_num = cell_size + 3
    adj = torch.zeros((node_num, node_num))

    # 遍历cell节点
    for i in range(cell_size):
        # 对每个连接选择使用Gumbel-Softmax进行采样，得到的是概率分布
        connect_probs = gumbel_softmax(alpha[i], temperature=temperature)

        # 假设我们将概率分布应用于直接连接到输出的节点
        # 以下逻辑确保特殊处理`j == 0`或`j == 1`的情况
        for j in range(node_num -1):  # 遍历可能的连接点
            if j == 0 or j == 1:
                adj[j, i + 2] = connect_probs[j]  # 特殊节点到当前cell节点的连接
            else:
                # 对于其他连接，我们简化逻辑，直接使用概率值
                # 注意：这里的实现需要根据实际场景调整
                # 可能的概率都要加上
                adj[j, i + 2] = connect_probs[j % len(connect_probs)]

    adj[2:-1, -1] = 1  # 假设每个cell节点直接连接到输出节点

    return adj

# 初始化alpha参数
cell_size = 8
num_possible_connections = 4
alpha = torch.randn((cell_size, num_possible_connections), requires_grad=True)

# 设置优化器
optimizer = optim.Adam([alpha], lr=0.01)

# 简单的训练循环
for step in range(100):
    optimizer.zero_grad()
    adj = transform_matrix_gumbel(alpha, cell_size, temperature=0.5)
    # 假设损失函数是adj矩阵所有元素的平方和
    loss = adj.pow(2).sum()
    loss.backward()
    optimizer.step()

    if step % 10 == 0:
        print(f"Step {step}, Loss: {loss.item()}")
        print(adj)
