from utils.data_loader import load_all_graphs

if __name__ == '__main__':
    graphs, graphs_num = load_all_graphs('./json')
    # 收集每个图出现的标签
    labels_list = [graph["labels"] for graph in graphs]

    one_hot_dim = 0

    # 统计每个图对出现的标签
    for i in range(0, graphs_num):
        for j in range(i, graphs_num):
            # 合并graph1和graph2的标签
            combined_labels = set()
            for label in graphs[i]["labels"]:
                if label not in combined_labels:
                    combined_labels.add(label)
            for label in graphs[j]["labels"]:
                if label not in combined_labels:
                    combined_labels.add(label)

            one_hot_dim = max(one_hot_dim, len(combined_labels))

    # 将统计结果写进文件
    with open('dimension.txt', 'w') as f:
        f.write(str(one_hot_dim))

    print(f'one hot dimension: {one_hot_dim}')


