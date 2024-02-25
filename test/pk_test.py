import torch


def cal_pk(num, pre, gt):
    tmp = list(zip(gt, pre))
    tmp.sort()
    beta = []
    for i, p in enumerate(tmp):
        beta.append((p[1], p[0], i))
    beta.sort()
    ans = 0
    for i in range(num):
        if beta[i][2] < num:
            ans += 1
    return ans / num


pre = torch.Tensor([1, 2, 3, 4, 5, 6])
gt = torch.Tensor([1, 2, 5, 4, 3, 6])

cal_pk(3, pre, gt)
