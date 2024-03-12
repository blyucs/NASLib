import logging
import torch
import os
import numpy as np
from naslib.defaults.predictor_evaluator import PredictorEvaluator
from naslib.utils.encodings import EncodingType
from naslib.search_spaces.nasbench301.encodings import OPS
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt
from naslib.predictors.gcn import NeuralPredictorModel
import glob

from naslib.predictors import (
    GCNPredictor,
)

from naslib.search_spaces.core.query_metrics import Metric
from naslib.search_spaces import (
    NasBench101SearchSpace,
    NasBench201SearchSpace,
    NasBench301SearchSpace,
    NasBenchNLPSearchSpace,
    TransBench101SearchSpaceMicro,
    TransBench101SearchSpaceMacro,
    NasBenchASRSearchSpace,
)
from naslib import utils
from naslib.utils import setup_logger, get_dataset_api, create_cpfile_dir

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

config = utils.get_config_from_args(config_type="predictor")
utils.set_seed(config.seed)
logger = setup_logger(config.save + "/log.log")
logger.setLevel(logging.INFO)
utils.log_args(config)
create_cpfile_dir(config.save, scripts_to_save=glob.glob('naslib/runners/predictors/*.py'))

supported_predictors = {
    "gcn": GCNPredictor(encoding_type=EncodingType.GCN, ss_type = "nasbench301", hpo_wrapper=True),
}

supported_search_spaces = {
    # "nasbench101": NasBench101SearchSpace(),
    # "nasbench201": NasBench201SearchSpace(),
    "nasbench301": NasBench301SearchSpace(),
    # "nlp": NasBenchNLPSearchSpace(),
    # 'transbench101_micro': TransBench101SearchSpaceMicro(config.dataset),
    # 'transbench101_macro': TransBench101SearchSpaceMacro(),
    # "asr": NasBenchASRSearchSpace(),
}

"""
If the API did not evaluate *all* architectures in the search space, 
set load_labeled=True
"""
load_labeled = True if config.search_space in ["nasbench301", "nlp"] else False
dataset_api = get_dataset_api(config.search_space, config.dataset)

# initialize the search space and predictor
utils.set_seed(config.seed)
predictor = supported_predictors[config.predictor]
search_space = supported_search_spaces[config.search_space]

# initialize the PredictorEvaluator class
# predictor_evaluator = PredictorEvaluator(predictor, config=config)
# predictor_evaluator.adapt_search_space(
#     search_space, load_labeled=load_labeled, dataset_api=dataset_api
# )

node_predecessors = [
    [0, 1, 2],                  # 节点3
    [0, 1, 2, 3],               # 节点4
    [0, 1, 2, 3, 4],            # 节点5
]

temp  = 1

def prepare_input():
    matrices = []
    ops = []

    # for cell in range(2):
    # normal cell sampling
    mat = transform_adj_matrix(normal_edge_index_alpha_dict)
    op = transform_op_matrix(normal_node_attr_alpha)
    matrices.append(mat)
    ops.append(op)

    # reduction cell sampling
    mat = transform_adj_matrix(reduction_edge_index_alpha_dict)
    op = transform_op_matrix(reduction_node_attr_alpha)
    matrices.append(mat)
    ops.append(op)

    mat_length = len(matrices[0][0])
    merged_length = len(matrices[0][0]) * 2
    matrix_final = torch.zeros((merged_length, merged_length))

    for col in range(mat_length):
        for row in range(col):
            matrix_final[row, col] = matrices[0][row, col]
            matrix_final[row + mat_length, col + mat_length] = matrices[1][row, col]

    # add 10 -> 11,12
    matrix_final[mat_length-1][mat_length] = 1
    matrix_final[mat_length-1][mat_length+1] = 1

    ops_onehot = torch.cat((ops[0], ops[1]), dim=0)

    # matrix_final = np.array(matrix_final, dtype=np.float32)
    # ops_onehot = np.array(ops_onehot, dtype=np.float32)
    num_vert = torch.tensor([22])
    val_acc = torch.tensor([0.0])
    dic = {
        "num_vertices": num_vert,
        "adjacency": matrix_final.unsqueeze(0),
        "operations": ops_onehot.unsqueeze(0),
        "val_acc": val_acc,
    }
    return dic

# cell_compact = (
#     ((0, 5), (1, 1), (0, 0), (1, 6), (1, 6), (3, 2), (0, 0), (2, 5)),   # NORMAL
#  ((0, 4), (1, 6), (1, 3), (2, 4), (0, 4), (3, 1), (2, 5), (4, 3))    # REDUCTION
# )

def transform_adj_matrix(alpha_dict):
    node_num = 11
    adj = torch.zeros((node_num, node_num))

    # 遍历cell节点
    for i in range(len(alpha_dict)):
        #[O4,O5],[O6,O7],[O8,O9]
        # 对每个连接选择使用Gumbel-Softmax进行采样，得到的是概率分布  是否采样gumbel + top-k ？
        connect_probs = F.softmax(alpha_dict[f'log_alpha_{i}'], dim = 0)
        # O4
        for j in node_predecessors[i][0:-1]:  # 遍历可能的连接点
            if j == 0 or j == 1:
                adj[j][2*i + 4] = connect_probs[j] # 特殊节点到当前cell节点的连接
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

def get_gumbel_dist(log_alpha):
    # log_alpha 2d one_hot 2d
    u = torch.zeros_like(log_alpha).uniform_()
    softmax = torch.nn.Softmax(-1)
    r = softmax((log_alpha + (-((-(u.log())).log()))) / temp)
    return r
def get_indices(r):
    # r_hard = (r == r.max(1, keepdim=True)[0]).float()  #  STE 能否解决 ？
    r_hard = torch.argmax(r, dim=1)
    # r_re = (r_hard - r).detach() + r
    r_hard_one_hot = F.one_hot(r_hard, num_classes=r.size(1)).float()  # 转换为 one-hot 编码，形状为 [batch_size, num_classes] 预留identity 和 output
    # 使用直通估计器（STE）的技巧
    r_re = (r_hard_one_hot - r).detach() + r
    return r_re

def transform_op_matrix(alpha_dict):
    pre_node = torch.zeros((2, len(OPS) + 2))
    pre_node[0:2, 0] = 1
    gumbel_distribution = get_gumbel_dist(alpha_dict)
    arch_indices = get_indices(gumbel_distribution)
    left_expanded = torch.cat([torch.zeros(8, 1), arch_indices], dim=1)
    expanded = torch.cat([left_expanded, torch.zeros(8, 1)], dim=1)
    tail_node = torch.zeros((1, len(OPS) + 2))
    tail_node[:,-1] = 1
    result = torch.cat([pre_node, expanded, tail_node], dim=0)
    return result


# 模型文件的路径
# model_path = os.path.join('/home/lvbo/00_code/NASLib/p301-0/cifar10/predictors/gcn/20240306-151033', 'surrogate_model.model')
model_path = os.path.join('/home/lvbo/00_code/NASLib/p301-0/cifar10/predictors/gcn/20240307-164851', 'surrogate_model.model')

# def get_model(ss_type):
#     if ss_type == "nasbench101":
#         initial_hidden = 5
#     elif ss_type == "nasbench201":
#         initial_hidden = 7
#     elif ss_type == "nasbench301":
#         initial_hidden = 9
#     elif ss_type == "nlp":
#         initial_hidden = 8
#     elif ss_type == "transbench101":
#         initial_hidden = 7
#     else:
#         raise NotImplementedError()
#
#     predictor = NeuralPredictorModel(initial_hidden=initial_hidden)
#     return predictor

model = predictor.get_model()
# 加载模型状态字典
model.load_state_dict(torch.load(model_path))
model.to(device)

# edge index distribution
normal_edge_index_alpha_dict = torch.nn.ParameterDict()
reduction_edge_index_alpha_dict = torch.nn.ParameterDict()

for node_idx, preds in enumerate(node_predecessors):
    # 创建一个Parameter对象，维度是该节点前序节点的数量
    log_alpha = Parameter(torch.zeros(len(preds)).normal_(1, 0.01), requires_grad=True)
    normal_edge_index_alpha_dict[f'log_alpha_{node_idx}'] = log_alpha

for node_idx, preds in enumerate(node_predecessors):
    # 创建一个Parameter对象，维度是该节点前序节点的数量
    log_alpha = Parameter(torch.zeros(len(preds)).normal_(1, 0.01), requires_grad=True)
    reduction_edge_index_alpha_dict[f'log_alpha_{node_idx}'] = log_alpha

# node attr distribution
normal_node_attr_alpha = Parameter(
    torch.zeros([8, len(OPS)]).normal_(1, 0.01), requires_grad=True
)

reduction_node_attr_alpha = Parameter(
    torch.zeros([8, len(OPS)]).normal_(1, 0.01), requires_grad=True
)

all_parameters = list(normal_edge_index_alpha_dict.parameters()) + [normal_node_attr_alpha] + \
                 list(reduction_edge_index_alpha_dict.parameters()) + [reduction_node_attr_alpha]
#
edge_parameters = list(normal_edge_index_alpha_dict.parameters()) + list(reduction_edge_index_alpha_dict.parameters())
node_parameters =  [normal_node_attr_alpha] + [reduction_node_attr_alpha]

class Temp_Scheduler(object):
    def __init__(self, total_epochs, curr_temp, base_temp, temp_min=0.33, last_epoch=-1):
        self.curr_temp = curr_temp
        self.base_temp = base_temp
        self.temp_min = temp_min
        self.last_epoch = last_epoch
        self.total_epochs = total_epochs
        self.step(last_epoch + 1)

    def step(self, epoch=None):
        return self.decay_whole_process()

    def decay_whole_process(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        # self.total_epochs = 150
        self.curr_temp = (1 - self.last_epoch / self.total_epochs) * (self.base_temp - self.temp_min) + self.temp_min
        if self.curr_temp < self.temp_min:
            self.curr_temp = self.temp_min
        return self.curr_temp

# 设置优化器
# optimizer = optim.Adam(log_edge_index_alpha_dict.parameters(), lr=0.01)
# optimizer = optim.Adam(all_parameters, lr=0.02)
optimizer = optim.Adam(node_parameters, lr=0.05)
# optimizer = optim.Adam(edge_parameters, lr=0.001)
base_temp = 1.0
min_temp = 0.03
epochs = 100
temp_scheduler = Temp_Scheduler(epochs, temp, base_temp,
                                temp_min=min_temp)
def reverse_transform_matrix(adj, ops):
    cell = []
    node_num = adj.shape[0] - 3  # 减去输入层和输出层的节点数
    # 对于成对出现的连接，确保每对只处理一次
    processed_pairs = set()

    for i in range(2, node_num + 2):
        for j in range(2, node_num + 2, 2):  # 以步长为2遍历，只考虑成对的连接起点
            if adj[j, i] == 1 and (j, i) not in processed_pairs:
                # 推导connect的值，这里j对应的是成对连接的第一个节点
                connect = (j - 2) // 2 + 2
                # 将这对连接的处理标记为已处理
                processed_pairs.add((j, i))
                processed_pairs.add((j + 1, i))  # 加1表示处理成对的第二个连接
                # 添加连接信息到cell中，这里假设没有操作信息
                # 确定操作，根据ops矩阵中的信息
                op_index = ops[i].argmax() - 1  # 假设ops矩阵中包含了一个额外的“无操作”类型在索引0
                cell.append((connect,op_index))

    return cell

# from alpha to compact
def get_compact_from_alpha(normal_edge_index_alpha_dict, reduction_edge_index_alpha_dict, normal_node_attr_alpha,
                           reduction_node_attr_alpha):
    def extract_cell(edge_alpha_dict, node_attr_alpha):
        cell = []
        # 处理from node 为0,1 的tuple
        for i in range(2):
            op_idx = node_attr_alpha[i].argmax().item()  # 假设前两个节点为输入节点，调整索引以匹配
            cell.append((i, op_idx))
        for edge_idx, alpha in enumerate(edge_alpha_dict.items()):
            # 获取最可能的操作索引
            top2_op_idx = alpha[1].topk(2).indices.tolist()
            # 从edge_key中提取节点索引
            for j, from_idx in enumerate(top2_op_idx):
                op_idx = node_attr_alpha[2*(edge_idx+1) + j].argmax().item()  # 假设前两个节点为输入节点，调整索引以匹配
                cell.append((from_idx, op_idx))

        return cell

    # 提取NORMAL和REDUCTION阶段的cell信息
    normal_cell = extract_cell(normal_edge_index_alpha_dict, normal_node_attr_alpha)
    reduction_cell = extract_cell(reduction_edge_index_alpha_dict, reduction_node_attr_alpha)

    # 将它们组合成compact形式
    cell_compact = (normal_cell, reduction_cell)

    return cell_compact

steps = []
accuracies = []
best_acc = 0.0
best_acc_array = []
valid_freq = 2
for step in range(epochs):
    optimizer.zero_grad()
    input_dic = prepare_input()
    pred = model(input_dic)
    loss = -pred.sum()
    loss.backward(retain_graph=True)
    optimizer.step()
    temp = temp_scheduler.step()

    if step % valid_freq == 0:
        compact = get_compact_from_alpha(normal_edge_index_alpha_dict, reduction_edge_index_alpha_dict,
                                         normal_node_attr_alpha, reduction_node_attr_alpha)
        arch = search_space.clone()
        arch.set_compact(compact)
        acc = arch.query(
            metric=Metric.VAL_ACCURACY, dataset='cifar10', dataset_api=dataset_api
        )
        if acc > best_acc:
            best_acc = acc

        print(f"Step {step}, Loss: {pred.item()}, acc: {acc}")
        print(f"Compact: {arch.get_compact()}")
        # 收集当前步骤数和准确率
        steps.append(step)
        accuracies.append(acc)
        best_acc_array.append(best_acc)

    if step == epochs-1:
        # print(normal_edge_index_alpha_dict.items())
        for name, param in normal_edge_index_alpha_dict.items():
            print(f"{name}: {param}")
        # print(reduction_edge_index_alpha_dict.items())
        for name, param in reduction_edge_index_alpha_dict.items():
            print(f"{name}: {param}")

        print(normal_node_attr_alpha)
        print(reduction_node_attr_alpha)
        print(f"search epochs: {epochs}, searched best arch's acc: {best_acc}")
        # print(input_dic['adjacency'])
        # print(input_dic['operations'])

# 在训练循环结束后绘制最佳准确率和当前准确率的对比图
plt.figure(figsize=(10, 6))

# 绘制当前准确率
plt.plot(steps, accuracies, marker='o', linestyle='-', color='b', label='Validation Accuracy')

# 绘制最佳准确率
plt.plot(steps, best_acc_array, marker='x', linestyle='--', color='r', label='Best Validation Accuracy')

plt.title('Optimization Curve of Accuracy')
plt.xlabel('Step')
plt.ylabel('Accuracy')
plt.legend(loc="upper left")
plt.grid(True)
plt.savefig(config.save + '/accuracy_curve.png')
plt.show()

# evaluate the predictor
# predictor_evaluator.evaluate()


