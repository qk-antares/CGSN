import os
import json


def convert_json(json_data):
    nodes = []
    edges = []

    n = json_data["n"]
    labels = json_data["labels"]
    graph = json_data["graph"]

    # Create nodes
    for i in range(n):
        node = {"id": str(i), "label": labels[i]}
        nodes.append(node)

    # Create edges
    for edge in graph:
        edge_entry = {"source": str(edge[0]), "target": str(edge[1])}
        edges.append(edge_entry)

    result = {"nodes": nodes, "edges": edges}
    return result


def convert_and_save_all(directory_path, output_directory):
    # 创建输出目录
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # 遍历目录下的所有文件
    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):
            file_path = os.path.join(directory_path, filename)

            # 读取原始JSON文件
            with open(file_path, 'r') as file:
                json_data = json.load(file)

            # 转换格式
            converted_result = convert_json(json_data)

            # 构建输出文件路径
            output_file_path = os.path.join(output_directory, f"{os.path.splitext(filename)[0]}.json")

            # 写入转换后的JSON数据到输出文件
            with open(output_file_path, 'w') as output_file:
                json.dump(converted_result, output_file, separators=(',', ':'))


# 示例用法
input_directory = "../data/AIDS_700/json"
output_directory = "../data/AIDS_700/out"

convert_and_save_all(input_directory, output_directory)
