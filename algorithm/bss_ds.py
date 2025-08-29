import random
import numpy as np
import torch
import torch.nn.functional as F
from model.cnn import CNN, TeacherCNN
from model.mlp import MLPNet, TeacherMLP
from algorithm.loss import loss_jocor
# import copy


# ------------------------- Noise Generateion ------------------------- #
def _generate_negative_labels(labels, num_classes, times):
    """Generate negative (incorrect) labels for consistency training."""
    labels = torch.flatten(labels)
    res = []
    for l in labels:
        others = random.sample([x for x in range(num_classes) if x != l.item()], times)
        res.extend(others)
    return torch.tensor(res).long()


def _repeat_images(samples, times):
    """Repeat each sample 'times' times for augmentation."""
    res = []
    for sample in samples:
        box = torch.cat([sample.unsqueeze(0)] * times, dim=0)
        res.append(box)
    return torch.cat(res, dim=0).float()


def generate_several_times(samples, labels, num_classes, times, device):
    """Generate noisy and clean samples for teacher training."""
    labels_noisy = _generate_negative_labels(labels, num_classes, times).to(device)
    noisy_or_clean = torch.LongTensor([1] * len(labels) + [0] * len(labels_noisy)).to(device)

    temp = _repeat_images(samples, times).to(device)
    samples = torch.cat((samples, temp), dim=0).to(device)
    labels = torch.cat([labels, labels_noisy], dim=0).to(device)

    # Shuffle
    idx = torch.randperm(len(samples))
    return samples[idx], labels[idx], noisy_or_clean[idx]



def check_given_samples(model, images, labels, p_threshold, indices, noise_or_not, device):
    """Filter samples into clean/noisy groups using teacher confidence."""
    model.eval()
    confids = model(images, labels).cpu()

    clean, noisy = [], []
    for i, c in enumerate(confids):
        if c <= p_threshold:
            noisy.append(i)
        else:
            clean.append(i)

    pure_ratio_clear = np.sum(noise_or_not[indices[clean]]) / float(len(clean)) if len(clean) > 0 else 0
    model.train()

    return (
        images[clean],
        labels[clean],
        images[noisy],
        labels[noisy],
        confids[clean].to(device),
        pure_ratio_clear,
        indices[clean],
    )


def loss_binary(consis_pred, consis, w_alpha):
    """Binary consistency loss for teacher model."""
    weights = consis.detach().clone().float()
    weights[weights == 1] = w_alpha
    weights[weights == 0] = 1 - w_alpha

    return F.binary_cross_entropy(
        consis_pred, consis.unsqueeze(-1).float(), weight=weights.unsqueeze(-1)
    )


def update(lamda, prev_params, model):
    """EMA-like update of teacher model parameters."""
    for param, prev in zip(model.parameters(), prev_params):
        param.data = (1 - lamda) * param.data + lamda * prev









class EMASmoother:
    """
    Exponential Moving Average (EMA) Smoother for model parameters.
    对 teacher 模型参数进行轨迹平滑，缓解训练不稳定。
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.99):
        self.decay = decay
        device = next(model.parameters()).device
        # 确保 shadow_params 跟随模型设备
        self.shadow_params = [p.data.clone().to(device) for p in model.parameters()]

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        """使用 EMA 更新 shadow_params，自动处理设备差异。"""
        for s_param, param in zip(self.shadow_params, model.parameters()):
            s_param.mul_(self.decay).add_(param.data.to(s_param.device), alpha=1 - self.decay)

    @torch.no_grad()
    def apply_to(self, model: torch.nn.Module):
        """将平滑后的参数写回模型"""
        for s_param, param in zip(self.shadow_params, model.parameters()):
            param.data.copy_(s_param.to(param.device))

    def state_dict(self):
        """保存 EMA 状态"""
        return {
            "shadow_params": [p.clone() for p in self.shadow_params],
            "decay": self.decay,
        }

    def load_state_dict(self, state_dict: dict):
        """恢复 EMA 状态"""
        self.decay = state_dict["decay"]
        # 保持 shadow_params 与模型设备一致
        self.shadow_params = [p.clone().to(self.shadow_params[0].device) for p in state_dict["shadow_params"]]








# ------------------------- BSS Main Class ------------------------- #
class BSS:
    def __init__(self, args, train_dataset, device, input_channel, num_classes):
        self.args = args
        self.device = device
        self.train_dataset = train_dataset
        self.num_classes = num_classes

        # Noise handling
        self.noise_or_not = train_dataset.noise_or_not
        forget_rate = args.noise_rate / 2 if args.noise_type == "asymmetric" else args.noise_rate

        # Optimizer scheduling
        lr, mom1, mom2 = args.lr, 0.9, 0.1
        self.alpha_plan = [lr] * args.n_epoch
        self.beta1_plan = [mom1] * args.n_epoch
        for i in range(args.epoch_decay_start, args.n_epoch):
            self.alpha_plan[i] = float(args.n_epoch - i) / (args.n_epoch - args.epoch_decay_start) * lr
            self.beta1_plan[i] = mom2

        # Forget rate scheduling
        self.rate_schedule = np.ones(args.n_epoch) * forget_rate
        self.rate_schedule[: args.num_gradual] = np.linspace(0, forget_rate ** args.exponent, args.num_gradual)

        # Training params
        self.co_lambda = args.co_lambda
        self.n_epoch = args.n_epoch
        self.early_stop = args.early_stop
        self.p_thresh = args.p_thresh
        self.s_thresh = args.s_thresh
        self.dataset = args.dataset
        # self.lambda_schedule = generate_lambda_schedule(self.early_stop, self.n_epoch, self.thresh)
        self.times = args.times
        self.print_freq = 50
        self.adjust_lr = 1

        # Models
        if args.model_type == "cnn":
            self.model1 = CNN(input_channel=input_channel, n_outputs=num_classes)
            self.model2 = CNN(input_channel=input_channel, n_outputs=num_classes)
            self.teacher = TeacherCNN(num_classes)
        elif args.model_type == "mlp":
            self.model1 = MLPNet()
            self.model2 = MLPNet()
            self.teacher = TeacherMLP(num_classes)
        else:
            raise ValueError(f"Unsupported model type: {args.model_type}")

        self.teacher_ema = EMASmoother(self.teacher, decay=0.95)
        
        self.model1.to(device)
        self.model2.to(device)
        self.teacher.to(device)

        # Optimizers
        self.optimizer = torch.optim.Adam(
            list(self.model1.parameters()) + list(self.model2.parameters()), lr=lr
        )
        self.teacher_optimizer = torch.optim.Adam(self.teacher.parameters(), lr=1e-3)

        # Loss function
        self.loss_fn = loss_jocor

    # ------------------------- Evaluation ------------------------- #
    def evaluate_model(self, model, test_loader):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels, _ in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                preds = torch.argmax(model(images), dim=1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        return 100.0 * correct / total

    def evaluate(self, test_loader):
        return self.evaluate_model(self.model1, test_loader), self.evaluate_model(self.model2, test_loader)

    # ------------------------- Training ------------------------- #
    def train(self, train_loader, epoch):
        self.model1.train()
        self.model2.train()
        self.teacher.train()

        if self.adjust_lr:
            self.adjust_learning_rate(epoch)

        stats = {"pure_ratio_1": [], "pure_ratio_2": [], "pure_ratio_check": [], "loss_cls": [], "loss_teacher": []}

        for i, (images, labels, indices) in enumerate(train_loader):
            indices = indices.cpu().numpy().transpose()
            images, labels = images.to(self.device), labels.to(self.device)

            # Teacher filtering
            if epoch > self.early_stop:
                images_clean, labels_clean, _, _, _, pure_ratio_check, idx_clean = check_given_samples(
                    self.teacher, images, labels, self.p_thresh, indices, self.noise_or_not, device=self.device
                )
            else:
                images_clean, labels_clean, pure_ratio_check, idx_clean = images, labels, 0, indices

            logits1, logits2 = self.model1(images_clean), self.model2(images_clean)

            # JoCoR loss
            loss_1, loss_2, pure_ratio_1, pure_ratio_2, imgs_sel, labels_sel = self.loss_fn(
                logits1,
                logits2,
                images_clean,
                labels_clean,
                self.s_thresh if epoch > self.early_stop else self.rate_schedule[epoch],
                idx_clean,
                self.noise_or_not,
                self.co_lambda,
            )
            if len(labels_sel) == 0:
                continue

            # Backprop (model1 & model2)
            self.optimizer.zero_grad()
            loss_1.backward()
            self.optimizer.step()

            # Teacher update
            imgs_bin, labels_bin, consis_bin = generate_several_times(imgs_sel, labels_sel, self.num_classes, self.times, self.device)

            preds_teacher = self.teacher(imgs_bin, labels_bin)
            loss_teacher = loss_binary(preds_teacher, consis_bin, self.args.w_alpha)

            self.teacher_optimizer.zero_grad()
            loss_teacher.backward()
            self.teacher_optimizer.step()

            # 更新 teacher EMA
            self.teacher_ema.update(self.teacher)
            self.teacher_ema.apply_to(self.teacher)  # 覆盖 teacher 参数为平滑后的版本

            # Log stats
            stats["pure_ratio_1"].append(100 * pure_ratio_1)
            stats["pure_ratio_2"].append(100 * pure_ratio_2)
            stats["pure_ratio_check"].append(100 * pure_ratio_check)
            stats["loss_cls"].append(loss_1.item())
            stats["loss_teacher"].append(loss_teacher.item())

            if i % self.print_freq == 0:
                print(
                    f"{self.dataset} | Epoch [{epoch:03d}] Batch {i:03d}/{len(train_loader)} | "
                    f"Sel Acc: {pure_ratio_check:.4f}, {pure_ratio_1:.4f} | "
                    f"CE Loss: {loss_1.item():.4f} | Teacher Loss: {loss_teacher.item():.4f}"
                )

        return (
            stats["pure_ratio_1"],
            # stats["pure_ratio_2"],
            stats["pure_ratio_check"],
            np.round(np.mean(stats["loss_cls"]), 4),
            np.round(np.mean(stats["loss_teacher"]), 4),
        )

    def adjust_learning_rate(self, epoch):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.alpha_plan[epoch]
            param_group["betas"] = (self.beta1_plan[epoch], 0.999)
