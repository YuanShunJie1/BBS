
from email.mime import image
import math
import torch 
import torch.nn.functional as F
import numpy as np
# from device import device
import torch.nn as nn
import random



def build_for_cifar100(size, noise):
    """ The noise matrix flips to the "next" class with probability 'noise'.
    """
    assert(noise >= 0.) and (noise <= 1.)

    P = (1. - noise) * np.eye(size)
    for i in np.arange(size - 1):
        P[i, i+1] = noise
    # adjust last row
    P[size-1, 0] = noise
    return P

def multiclass_noisify(y, P, random_state=0):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """
    m = y.shape[0]
    # print(m)
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = np.array([y[idx]])
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, P[i, :][0], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y

def generate_noisy_transition_matrix(noise_type, dataset, nb_classes, random_state=0):

    P = np.ones((nb_classes, nb_classes))
    n = 1
    P = (n / (nb_classes - 1)) * P

    # 0 -> 1
    P[0, 0] = 1. - n
    for i in range(1, nb_classes-1):
        P[i, i] = 1. - n
    P[nb_classes-1, nb_classes-1] = 1. - n

    return P

def merge(images_clean, labels_clean, images_noisy, labels_noisy_corrected, confids_clean, confids_noisy_corrected):
    images = torch.cat([images_clean, images_noisy], dim=0)
    labels = torch.cat([labels_clean, labels_noisy_corrected], dim=0)
    confids = torch.cat([confids_clean, confids_noisy_corrected], dim=0)

    return images, labels, confids


# def _generate_2(labels, num_classes, times):
#     labels = torch.flatten(labels)
#     res = torch.tensor([])
#     for l in labels:
#         others = torch.tensor(random.sample([x for x in range(num_classes) if x != l.item()], times))
#         res = torch.cat([res, others], dim=0)
#     return res.long()

# def _concat_images(samples, times):
#     res = None
#     for sample in samples:
#         box = torch.cat([sample.unsqueeze(0)] * times, dim=0)
#         if res == None:
#             res = box
#         else:
#             res = torch.cat([res, box], dim=0)
#     return res.float()

# def generate_several_times(samples, labels, num_classes, times, device):
#     labels_noisy = _generate_2(labels, num_classes, times).to(device)
#     noisyOrclean = torch.LongTensor([1] * len(labels) + [0] * len(labels_noisy)).to(device)
#     temp = _concat_images(samples, times).to(device)
#     samples = torch.cat((samples, temp), dim=0).to(device)
#     labels = torch.cat([labels, labels_noisy], dim=0).to(device)

#     idx = torch.randperm(len(samples))
#     samples = samples[idx].to(device)
#     labels = labels[idx].to(device)
#     noisyOrclean = noisyOrclean[idx].to(device)

#     return samples, labels, noisyOrclean


# # modelc, images, labels, args.P_threshold, ind, noise_or_not
# def check_given_samples(model, images, labels, p_threshold, ind, noise_or_not, device):
#     model.eval()
#     confids = model(images, labels).cpu()

#     clean = []
#     noisy = []
#     # 0 noisy
#     # 1 clean 
#     for i in range(len(images)):
#         if confids[i] <= p_threshold:
#             noisy.append(i)
#         else:
#             clean.append(i)

#     pure_ratio_clear = np.sum(noise_or_not[ind[clean]]) / float(len(clean))
#     model.train()
#     return images[clean], labels[clean], images[noisy], labels[noisy], confids[clean].to(device), pure_ratio_clear, ind[clean]




# def loss_binary(consis_pred, consis, w_alpha):
#     weights = consis.detach().clone().float()
#     weights[weights == 1] = w_alpha
#     weights[weights == 0] = 1 - w_alpha

#     consis = consis.unsqueeze(-1)
#     weights = weights.unsqueeze(-1)
#     loss = F.binary_cross_entropy(consis_pred, consis.float(), weight=weights)
#     return loss


# # Loss functions
# def loss_coteaching(y_1, y_2, labels, forget_rate, images, ind, noise_or_not):

#     loss_1 = F.cross_entropy(y_1, labels, reduction = 'none')
#     ind_1_sorted = np.argsort(loss_1.cpu().data)

#     loss_2 = F.cross_entropy(y_2, labels, reduction = 'none')
#     ind_2_sorted = np.argsort(loss_2.cpu().data)

#     num_remember = int((1 - forget_rate) * len(labels))

#     ind_1_update=ind_1_sorted[:num_remember]
#     ind_2_update=ind_2_sorted[:num_remember]
    
#     # exchange
#     loss_1_update = F.cross_entropy(y_1[ind_2_update], labels[ind_2_update])
#     loss_2_update = F.cross_entropy(y_2[ind_1_update], labels[ind_1_update])   


#     pure_ratio_1 = np.sum(noise_or_not[ind[ind_1_update]])/float(num_remember)
#     pure_ratio_2 = np.sum(noise_or_not[ind[ind_2_update]])/float(num_remember)

#     all_inds = list(set(ind_1_update.tolist() + ind_2_update.tolist()))

#     loss1 = torch.sum(loss_1_update)/num_remember
#     loss2 = torch.sum(loss_2_update)/num_remember

#     return loss1, loss2, images[all_inds], labels[all_inds], pure_ratio_1, pure_ratio_2


# def update(lamda, params_box, model):
#     pb = params_box
#     index = 0
#     for param in model.parameters():
#         param.data = nn.parameter.Parameter((1 - lamda) * param.data + lamda * pb[index]) 
#         index = index + 1


# def generate_lambda_schedule(early_stop, epoches, thresh):
#     es = early_stop
#     def f_x(x, p):
#         return (thresh) / (1 + math.exp(-0.1 * (x - p)))
#     return [f_x(e, es + 5) for e in range(epoches)]




