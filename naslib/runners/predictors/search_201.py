import logging
import torch
import os
import numpy as np
from naslib.defaults.predictor_evaluator import PredictorEvaluator
from naslib.utils.encodings import EncodingType
from naslib.search_spaces.nasbench201.encodings import OPS
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
    "gcn": GCNPredictor(encoding_type=EncodingType.GCN, hpo_wrapper=True),
}

supported_search_spaces = {
    # "nasbench101": NasBench101SearchSpace(),
    "nasbench201": NasBench201SearchSpace(),
    # "nasbench301": NasBench301SearchSpace(),
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
    """
    Input:
    a list of categorical ops starting from 0
    """

    ops_onehot = transform_op_matrix(node_attr_alpha)
    matrix = torch.tensor(np.array(
        [
            [0, 1, 1, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    ))
    # matrix = np.transpose(matrix)
    dic = {
        "num_vertices": torch.tensor([8]),
        "adjacency": matrix,
        "operations": ops_onehot,
        "mask": torch.tensor(np.array([i < 8 for i in range(8)], dtype=np.float32)),
        "val_acc": torch.tensor([0.0]),
    }

    return dic

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
    pre_node = torch.zeros((1, len(OPS) + 2))
    pre_node[0, 0] = 1
    gumbel_distribution = get_gumbel_dist(alpha_dict)
    arch_indices = get_indices(gumbel_distribution)
    left_expanded = torch.cat([torch.zeros(6, 1), arch_indices], dim=1)
    expanded = torch.cat([left_expanded, torch.zeros(6, 1)], dim=1)
    tail_node = torch.zeros((1, len(OPS) + 2))
    tail_node[:,-1] = 1
    result = torch.cat([pre_node, expanded, tail_node], dim=0)
    return result


# 模型文件的路径
model_path = os.path.join('/home/lvbo/00_code/NASLib/p201-0/cifar10/predictors/gcn/single/20240311-165636', 'surrogate_model.model')

model = predictor.get_model("nasbench301")
# 加载模型状态字典
model.load_state_dict(torch.load(model_path))
model.to(device)

# node attr distribution
node_attr_alpha = Parameter(
    torch.zeros([6, len(OPS)]).normal_(1, 0.01), requires_grad=True
)

all_parameters = [node_attr_alpha]

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
optimizer = optim.Adam(all_parameters, lr=0.02)
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


