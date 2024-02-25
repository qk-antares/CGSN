import numpy as np
import matplotlib.pyplot as plt

from utils.data_loader import load_all_graphs, load_ged

graphs, _ = load_all_graphs('../AIDS_700/json')
ged_dict = load_ged('./TaGED.json')

sim_scores = []

for i in range(0, 700):
    for j in range(i, 700):
        gid1, gid2 = graphs[i]['gid'], graphs[j]['gid']
        n1, n2 = graphs[i]['n'], graphs[j]['n']
        gid_pair = (gid1, gid2)
        if gid_pair not in ged_dict:
            gid_pair = (gid2, gid1)
        sim_score = np.exp(-2 * ged_dict[gid_pair][0][0] / (n1 + n2))
        sim_scores.append(sim_score)

plt.hist(sim_scores, bins=100, density=True, alpha=0.7, color='blue', edgecolor='black')
plt.title('Distribution of SimScore')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.savefig("./result.jpg")

static_pre = 0.37
mse = np.mean((static_pre-np.array(sim_scores))**2)
# 15.348e-3
print("模型恒预测SimScore=0.39时的MSE：", mse)
# 15.228e-3
print("模型恒预测SimScore=0.38时的MSE：", mse)
# 15.309e-3
print("模型恒预测SimScore=0.37时的MSE：", mse)
