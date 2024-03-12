

def transform_matrix(cell):

    node_num = len(cell) + 3

    adj = np.zeros((node_num, node_num))

    for i in range(len(cell)):
        connect,op = cell[i]
        if connect == 0 or connect == 1:
            adj[connect][i + 2] = 1
        else:
            adj[(connect - 2) * 2 + 2][i + 2] = 1
            adj[(connect - 2) * 2 + 3][i + 2] = 1
    adj[2:-1, -1] = 1
    return adj
    

    ops = np.zeros((node_num, len(OPS) + 2))
    ops[i + 2][op] = 1
    ops[0:2, 0] = 1
    ops[-1][-1] = 1
    
    

# Example usage
cell = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8)]
adj = transform_matrix(cell)
	
	
	import numpy as np
import torch
import torch.nn.functional as F

def gumbel_softmax(logits, temperature=1.0):
    """
    Apply the Gumbel-Softmax trick to 'logits' at a given 'temperature'.
    """
    gumbels = -torch.log(-torch.log(torch.rand_like(logits)))
    y_soft = logits + gumbels
    return F.softmax(y_soft / temperature, dim=-1)

def transform_matrix(cell, temperature=1.0):
    """
    Convert 'cell' into a differentiable adjacency matrix 'adj' using Gumbel-Softmax.
    """
    # Convert cell to tensor
    cell = torch.tensor(cell, dtype=torch.float32)
    
    # Extract 'connect' as logits for Gumbel-Softmax
    connect_logits = cell[:, 0].unsqueeze(1)  # We ignore 'op' for simplicity
    
    # Apply Gumbel-Softmax to get a differentiable approximation of 'connect'
    soft_connect = gumbel_softmax(connect_logits, temperature=temperature)
    
    # Initialize the adjacency matrix with zeros
    node_num = len(soft_connect) + 3
    adj = torch.zeros((node_num, node_num))
    
    # Use soft connections to fill in the adjacency matrix
    for i, soft_con in enumerate(soft_connect):
        adj[:2, i + 2] = soft_con[:2]  # Connections from input nodes
        # Connections from intermediate nodes, assuming binary decision
        adj[2 + i * 2:4 + i * 2, i + 2] = soft_con[2:].repeat(2)
    
    # The last node is connected to all intermediate nodes
    adj[2:-1, -1] = 1
    
    return adj

# Example usage
cell = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8)]
adj = transform_matrix(cell)



import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def transform_matrix_soft(cell, temperature=0.1):
    node_num = len(cell) + 3
    adj = np.zeros((node_num, node_num))
    
    # 假设有一个连接概率向量，对于每个cell中的连接，我们使用softmax来获取一个概率分布
    for i in range(len(cell)):
        connect, _ = cell[i]
        
        # 为了简化，这里我们仅仅考虑两个连接点的情况
        # 在实际应用中，你可能会有一个更复杂的连接概率向量
        connect_probs = softmax(np.array([1.0 if j == connect else 0.0 for j in range(node_num)]))
        
        # 使用softmax概率来加权连接强度
        for j in range(node_num):
            adj[j][i + 2] = connect_probs[j]

    # 考虑最后一层的连接
    adj[2:-1, -1] = softmax(np.ones(node_num-3))
    
    return adj




import torch

def gumbel_softmax(logits, temperature=1.0, dim=-1):
    gumbels = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)  # Gumbel分布
    y = logits + gumbels
    return torch.softmax(y / temperature, dim=dim)

def transform_matrix_gumbel(cell, temperature=0.1):
    node_num = len(cell) + 3
    adj = torch.zeros((node_num, node_num))

    for i, (connect, _) in enumerate(cell):
        logits = torch.zeros(node_num)
        logits[connect] = 1.0  # 假定连接点的logits为1，其他为0，模拟“硬”选择

        # 应用Gumbel-Softmax来得到软化的连接概率
        connect_probs = gumbel_softmax(logits, temperature=temperature)

        # 填充邻接矩阵
        if connect == 0 or connect == 1:
            adj[connect, i + 2] = connect_probs[connect]
        else:
            adj[(connect - 2) * 2 + 2, i + 2] = connect_probs[(connect - 2) * 2 + 2]
            adj[(connect - 2) * 2 + 3, i + 2] = connect_probs[(connect - 2) * 2 + 3]

    # 对最后一层的节点进行连接
    adj[2:-1, -1] = gumbel_softmax(torch.ones(node_num-3), temperature=temperature)[2:-1]

    return adj
    
    

# 临接矩阵不需要构造成one-hot 类型，而是描述一种概率， 这个概率来源于gumbel 采样的结果，其描述了一条边存在的一种概率。 


import torch

def gumbel_softmax(logits, temperature=1.0, dim=-1):
    gumbels = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)  # Gumbel分布
    y = logits + gumbels
    return torch.softmax(y / temperature, dim=dim)

def transform_matrix_gumbel(cell, temperature=0.1):
    node_num = len(cell) + 3
    adj = torch.zeros((node_num, node_num))

    for i, (connect, _) in enumerate(cell):
        logits = torch.zeros(node_num)
        logits[connect] = 1.0  # 假定连接点的logits为1，其他为0，模拟“硬”选择

        # 应用Gumbel-Softmax来得到软化的连接概率
        connect_probs = gumbel_softmax(logits, temperature=temperature)

        # 填充邻接矩阵
        if connect == 0 or connect == 1:
            adj[connect, i + 2] = connect_probs[connect]
        else:
            adj[(connect - 2) * 2 + 2, i + 2] = connect_probs[(connect - 2) * 2 + 2]
            adj[(connect - 2) * 2 + 3, i + 2] = connect_probs[(connect - 2) * 2 + 3]

    # 对最后一层的节点进行连接
    adj[2:-1, -1] = gumbel_softmax(torch.ones(node_num-3), temperature=temperature)[2:-1]

    return adj




