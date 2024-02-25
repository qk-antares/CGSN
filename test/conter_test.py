from collections import Counter

# 假设给定的元素字符串
g1 = {"n": 10, "m": 10, "labels": ["C", "O", "O", "C", "C", "C", "O", "C", "C", "C"], "graph": [[7, 4], [7, 9], [3, 0], [3, 5], [3, 4], [5, 8], [8, 9], [6, 4], [1, 0], [0, 2]]}
g2 = {"n": 6, "m": 6, "labels": ["S", "C", "N", "S", "C", "C"], "graph": [[3, 5], [3, 1], [5, 4], [1, 0], [1, 2], [4, 2]]}

# 使用 Counter 统计每个元素出现的次数
counter = Counter(g1["labels"] + g2["labels"])

# 按照出现次数从大到小排序
sorted_elements = sorted(counter.items(), key=lambda x: x[1], reverse=True)

# 创建元素及其索引的映射字典
element_index_map = {element: index for index, (element, _) in enumerate(sorted_elements)}
print(element_index_map)
