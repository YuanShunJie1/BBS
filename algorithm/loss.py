import torch 
import torch.nn.functional as F
import numpy as np
import torch.nn.functional as F



def kl_loss_compute(pred, soft_targets, reduce=True):
    log_probs = F.log_softmax(pred, dim=1)
    probs = F.softmax(soft_targets, dim=1)
    
    probs = torch.clamp(probs, min=1e-8)

    kl = F.kl_div(log_probs, probs, reduction='none')  # PyTorch 2.x 用 'none' 代替 reduce=False

    if reduce:
        return torch.mean(torch.sum(kl, dim=1))
    else:
        return torch.sum(kl, dim=1)


# def loss_jocor(y_1, y_2, images, labels, forget_rate, ind, noise_or_not, co_lambda=0.1):
#     # 交叉熵
#     loss_1 = F.cross_entropy(y_1, labels, reduction='none') * (1 - co_lambda)
#     loss_2 = F.cross_entropy(y_2, labels, reduction='none') * (1 - co_lambda)

#     # KL 散度
#     a = kl_loss_compute(y_1, y_2, reduce=False)
#     b = kl_loss_compute(y_2, y_1, reduce=False)

#     # 联合损失
#     loss_pick = loss_1 + loss_2 + co_lambda * (a + b)

#     # 样本选择
#     ind_sorted = torch.argsort(loss_pick.detach())
#     num_remember = max(1, int((1 - forget_rate) * len(ind_sorted)))
#     ind_update = ind_sorted[:num_remember]
    
#     ind_update = ind_update.detach().cpu().numpy()

#     pure_ratio = noise_or_not[ind[ind_update]].sum().item() / float(num_remember)

#     # 最终损失
#     loss = loss_pick[ind_update].mean()

#     return loss, loss, pure_ratio, pure_ratio, images[ind_update], labels[ind_update]


def loss_jocor(y_1, y_2, images, labels, forget_rate, ind, noise_or_not, co_lambda=0.1):
    # 交叉熵
    loss_1 = F.cross_entropy(y_1, labels, reduction='none') * (1 - co_lambda)
    loss_2 = F.cross_entropy(y_2, labels, reduction='none') * (1 - co_lambda)

    # KL 散度
    a = kl_loss_compute(y_1, y_2, reduce=False)
    b = kl_loss_compute(y_2, y_1, reduce=False)

    # 联合损失
    loss_pick = loss_1 + loss_2 + co_lambda * (a + b)

    # 样本选择
    ind_sorted = torch.argsort(loss_pick.detach())
    num_remember = max(1, int((1 - forget_rate) * len(ind_sorted)))
    ind_update = ind_sorted[:num_remember]
    ind_update = ind_update.detach().cpu().numpy()

    # Selection Accuracy (原版)
    pure_ratio = noise_or_not[ind[ind_update]].sum().item() / float(num_remember)

    # ---- Precision / Recall / F1 计算 ----
    selected = noise_or_not[ind[ind_update]]  # 选中的标签（1=干净, 0=噪声）
    TP = selected.sum().item()
    FP = len(selected) - TP
    total_clean = noise_or_not[ind].sum().item()  # batch中干净总数
    FN = total_clean - TP

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    # 最终损失
    loss = loss_pick[ind_update].mean()

    return (
        loss, loss, pure_ratio, pure_ratio,
        images[ind_update], labels[ind_update],
        precision, recall, f1, ind_update
    )