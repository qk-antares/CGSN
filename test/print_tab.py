from texttable import Texttable

# 创建 Texttable 对象
table = Texttable()

# 设置表头
table.add_row(["model_name", "dataset", "graph_set", "testing_pairs", "time_usage(s/100p)",
               "mse", "mae", "acc", "fea", "rho", "tau", "pk10", "pk20"])

# 添加第一行数据
table.add_row(["SimGNN", "AIDS_700", "test", "14000", "0.028",
               "2.696", "0.877", "0.349", "0.622", "0.829", "0.692", "0.624", "0.736"])

table.set_max_width(1000)

print(table.draw())