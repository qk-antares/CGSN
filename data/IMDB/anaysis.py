# 获取最大节点数
from utils.data_loader import load_all_graphs

def get_max_nodes():
    graphs, _ = load_all_graphs('./json')
    nodes = [graph['n'] for graph in graphs]
    print(max(nodes))

get_max_nodes()
