from concurrent.futures import ProcessPoolExecutor

import networkx as nx

import json
from utils.data_loader import load_all_graphs


def data_converter(graph):
    # 创建一个空的图
    g = nx.Graph()

    # 将节点添加到图
    for index, label in enumerate(graph["labels"]):
        node_id = index
        node_attrs = {'label': label}
        g.add_node(node_id, **node_attrs)

    # 将边添加到图
    g.add_edges_from(graph["graph"])
    return g


def node_match(node1, node2):
    return node1['label'] == node2['label']


def edge_match(edge1, edge2):
    return True


def process_data(graph1, graph2):
    """
    返回图对的编辑距离和所有的节点映射
    :param graph1:
    :param graph2:
    :return:
    """
    g1 = data_converter(graph1)
    g2 = data_converter(graph2)
    # 计算两个图之间的编辑距离
    paths, ged = nx.optimal_edit_paths(g1, g2, node_match=node_match, edge_match=edge_match)
    node_mappings = []
    for path in paths:
        node_mapping = [-1] * graph1["n"]
        for match in path[0]:
            if match[0] is not None:
                node_mapping[match[0]] = match[1]
        node_mappings.append(node_mapping)

    return int(ged), node_mappings


def save_result(chunk, result_file, progress_file, processed_count, gid1, gid2, ged, node_mappings):
    # 追加结果到 JSON 文件
    with open(result_file, 'a') as f:
        json.dump([gid1, gid2, ged, node_mappings], f, separators=(',', ':'))
        f.write(',\n')

    # 更新已处理的数据数量到文本文件
    with open(progress_file, 'w') as f:
        f.write(str(processed_count))

    print(f"Chunk: {chunk}. Processed graph_pair: {gid1}, {gid2}. Total processed: {processed_count}")


def get_process_graph_pair(chunk, chunk_size, graphs_num, processed_count):
    """
    求当前求解到的graph1，graph2的index
    :param chunk: 数据分为多少块
    :param chunk_size: 每块大小
    :param graphs_num: 数据集有多少个图
    :param processed_count: 当前chunk已经处理的图对数量
    :return:
    """
    processed_count = chunk * chunk_size + processed_count
    for i in range(0, graphs_num):
        # 这个循环中图对的个数
        pair_count = graphs_num-i
        if processed_count >= pair_count:
            processed_count -= pair_count
        else:
            return i, i + processed_count


def process_data_chunk(chunk, chunk_size, graphs):
    result_file = f"result/result{chunk}.json"
    progress_file = f"result/progress{chunk}.txt"
    graphs_num = len(graphs)

    # 读取当前已经处理的数据数量
    try:
        with open(progress_file, 'r') as f:
            processed_count = int(f.read())
    except (FileNotFoundError, ValueError):
        processed_count = 0

    if processed_count == chunk_size:
        return

    start_graph1, start_graph2 = get_process_graph_pair(chunk, chunk_size, graphs_num, processed_count)

    for i in range(start_graph1, graphs_num):
        for j in range(start_graph2, graphs_num):
            graph1, graph2 = graphs[i], graphs[j]
            # 确保节点数较少的那个图位于左侧
            if graph1["n"] > graph2["n"]:
                graph1, graph2 = graph2, graph1

            # 处理数据
            ged, node_mappings = process_data(graph1, graph2)
            processed_count += 1
            # 保存结果
            save_result(chunk, result_file, progress_file, processed_count,
                        graph1['gid'], graph2['gid'], ged, node_mappings)
            if processed_count == chunk_size:
                return
        start_graph2 = i + 1


def main():
    # 待处理的数据
    graphs, graphs_num = load_all_graphs('./json')
    graphs_num = len(graphs)

    # 结果分成175个文件存储（总结果可能有6G，分开存储方便查看），每个文件存储1402个图对的处理结果
    num_chunks = 175
    chunk_size = (graphs_num + 1) * graphs_num // 2 // num_chunks
    # 线程池大小
    max_workers = 12

    # 创建线程池
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 循环提交任务
        for chunk in range(0, num_chunks):
            # 提交任务到线程池
            executor.submit(process_data_chunk, chunk, chunk_size, graphs)


if __name__ == "__main__":
    main()
