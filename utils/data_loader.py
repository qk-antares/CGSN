import json
from glob import glob
from os.path import basename

import networkx as nx


def load_labels(path):
    global_labels = json.load(open(f'{path}/labels.json', 'r'))
    features = iterate_get_graphs(f'{path}/onehot', "onehot")
    return global_labels, features


def one_hot_encode_labels(graph1, graph2, one_hot_dim):
    """
    将两个图的labels属性转成one-hot编码（需要保证graph1和graph2的labels类型总和不超过模型的处理范围）
    :param graph1:
    :param graph2:
    :param one_hot_dim: 初始独热编码的维度
    :return:
    """
    # 合并graph1和graph2的标签
    combined_labels = []
    # 创建一个集合用于跟踪已经添加的元素
    added_labels = set()
    # 分别遍历g1和g2
    for label in graph1["labels"]:
        if label not in added_labels:
            combined_labels.append(label)
            added_labels.add(label)
    for label in graph2["labels"]:
        if label not in added_labels:
            combined_labels.append(label)
            added_labels.add(label)

    # 创建标签到索引的映射
    label_to_index = {label: i for i, label in enumerate(combined_labels)}

    graph1_one_hot = [[0] * one_hot_dim for _ in range(len(graph1['labels']))]
    graph2_one_hot = [[0] * one_hot_dim for _ in range(len(graph2['labels']))]

    # 对graph1和graph2进行one-hot编码
    for i, label in enumerate(graph1['labels']):
        index = label_to_index[label]
        graph1_one_hot[i][index] = 1
    for i, label in enumerate(graph2['labels']):
        index = label_to_index[label]
        graph2_one_hot[i][index] = 1

    return graph1_one_hot, graph2_one_hot


def load_ged(filepath):
    """
    加载每个图对的GED
    :param filepath: 保存GED的json
    :return:
    """
    ged_dict = dict()
    TaGED = json.load(open(filepath, 'r'))
    for (id_1, id_2, ged_value, ged_nc, ged_in, ged_ie, mappings) in TaGED:
        ta_ged = (ged_value, ged_nc, ged_in, ged_ie)
        ged_dict[(id_1, id_2)] = (ta_ged, mappings)
    return ged_dict


def load_one_hot_dim(path):
    # 读取one_hot_dim
    try:
        with open(f'{path}/dimension.txt', 'r') as f:
            one_hot_dim = int(f.read())
    except (FileNotFoundError, ValueError):
        one_hot_dim = 0
    return one_hot_dim


def load_all_graphs(path):
    """
    加载某个目录下的所有图
    :param path: 目录
    :return:
    """
    graphs = iterate_get_graphs(path, "json")
    graphs_num = len(graphs)
    return graphs, graphs_num


def iterate_get_graphs(path, file_format):
    """
    读取某个文件夹下的所有图数据
    :param path: Input path.
    :param file_format: The suffix name of required files.
    :return graphs: Networkx (dict) graphs.
    """
    assert file_format in ['gexf', 'json', 'onehot', 'anchor']
    graphs = []
    for file in get_file_paths(path, file_format):
        gid = int(basename(file).split('.')[0])
        if file_format == 'gexf':
            g = nx.read_gexf(file)
            g.graph['gid'] = gid
            if not nx.is_connected(g):
                raise RuntimeError('{} not connected'.format(gid))
        elif file_format == 'json':
            g = json.load(open(file, 'r'))
            g['gid'] = gid
        elif file_format in ['onehot']:
            g = json.load(open(file, 'r'))
        else:
            raise RuntimeError('Not supported file format: {}'.format(file_format))
        graphs.append(g)
    return graphs


def get_file_paths(path, file_format='json'):
    """
    返回排序后的文件路径列表
    :param path: 存放图数据的文件夹路径.
    :param file_format: The suffix name of required files.
    :return paths: The paths of all required files.
    """
    path = path.rstrip('/')
    paths = sorted_nicely(glob(path + '/*.' + file_format))
    return paths


def sorted_nicely(file_names):
    """
    对文件名进行排序
    :param file_names: A list of file names:str.
    :return: A nicely sorted file name list.
    """

    def try_int(s):
        try:
            return int(s)
        except ValueError:
            return s

    import re

    def alphanum_key(s):
        return [try_int(c) for c in re.split('([0-9]+)', s)]

    return sorted(file_names, key=alphanum_key)
