import random
import numpy as np
import torch
import torch.nn.functional as F
from model.cnn import CNN, TeacherCNN
from model.mlp import MLPNet, TeacherMLP
from algorithm.loss import loss_jocor
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import os
import seaborn as sns


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

    @torch.no_grad()
    def forward_with_shadow(self, model: torch.nn.Module, x: torch.Tensor):
        """使用 shadow_params 做前向推理，不改变原模型参数"""
        # 保存原参数
        orig_params = [p.data.clone() for p in model.parameters()]
        # 用 EMA 参数替换
        for s_param, param in zip(self.shadow_params, model.parameters()):
            param.data.copy_(s_param.to(param.device))
        # 前向
        out = model(x)
        # 恢复原参数
        for param, orig in zip(model.parameters(), orig_params):
            param.data.copy_(orig)
        return out




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
        self.times = args.times
        self.print_freq = 50
        self.adjust_lr = 1

        # Models
        if args.model_type == "cnn":
            self.student = CNN(input_channel=input_channel, n_outputs=num_classes)
            self.teacher = TeacherCNN(num_classes)
        elif args.model_type == "mlp":
            self.student = MLPNet()
            self.teacher = TeacherMLP(num_classes)
        else:
            raise ValueError(f"Unsupported model type: {args.model_type}")

        # EMA
        self.teacher_ema = EMASmoother(self.teacher, decay=0.90)
        self.student_ema = EMASmoother(self.student, decay=0.99)
        
        self.student.to(device)
        self.teacher.to(device)

        # Optimizers
        self.optimizer = torch.optim.Adam(self.student.parameters(), lr=lr)
        self.teacher_optimizer = torch.optim.Adam(self.teacher.parameters(), lr=1e-3)

        # Loss function
        self.loss_fn = loss_jocor

    # ------------------------- Training ------------------------- #
    def train(self, train_loader, epoch):
        self.student.train()
        self.teacher.train()

        if self.adjust_lr:
            self.adjust_learning_rate(epoch)

        stats = {"pure_ratio_student": [], "pure_ratio_check": [], "loss_student": [], "loss_teacher": [], "ps":[], "rs": [], "fs": []}

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

            # ---------------- Student vs Student EMA ----------------
            logits_student = self.student(images_clean)
            with torch.no_grad():
                logits_student_ema = self.student_ema.forward_with_shadow(self.student, images_clean)

            # Student loss (对比 Student vs Student EMA)
            loss_student, _, pure_ratio_1, pure_ratio_2, images_sec_sel, labels_sec_sel, precision, recall, f1, _ = self.loss_fn(
                logits_student,
                logits_student_ema,
                images_clean,
                labels_clean,
                self.s_thresh if epoch > self.early_stop else self.rate_schedule[epoch],
                idx_clean,
                self.noise_or_not,
                self.co_lambda,
            )

            if len(labels_sec_sel) == 0: continue
            
            self.optimizer.zero_grad()
            loss_student.backward()
            self.optimizer.step()

            # 更新 student EMA
            self.student_ema.update(self.student)

            # ---------------- Teacher ----------------
            imgs_bin, labels_bin, consis_bin = generate_several_times(images_sec_sel, labels_sec_sel, self.num_classes, self.times, self.device)
            preds_teacher = self.teacher(imgs_bin, labels_bin)
            loss_teacher = loss_binary(preds_teacher, consis_bin, self.args.w_alpha)

            self.teacher_optimizer.zero_grad()
            loss_teacher.backward()
            self.teacher_optimizer.step()

            # 更新 teacher EMA
            self.teacher_ema.update(self.teacher)
            self.teacher_ema.apply_to(self.teacher)

            # Log stats
            stats["pure_ratio_check"].append(100 * pure_ratio_check)
            stats["pure_ratio_student"].append(100 * pure_ratio_1)
            stats["loss_student"].append(loss_student.item())
            stats["loss_teacher"].append(loss_teacher.item())

            stats["ps"].append(precision)
            stats["rs"].append(recall)
            stats["fs"].append(f1)

            if i % self.print_freq == 0:
                print(
                    f"Dataset: {self.dataset} | Epoch [{epoch:03d}] Batch {i:03d}/{len(train_loader):03d} | "
                    f"TS-Acc: {stats['pure_ratio_check'][-1]:6.2f} | "
                    f"SS-Acc: {stats['pure_ratio_student'][-1]:6.2f} | "
                    f"CE-Loss: {loss_student.item():7.4f} | BCE-Loss: {loss_teacher.item():7.4f}"
                )

        return (
            stats["pure_ratio_student"],
            stats["pure_ratio_check"],
            np.round(np.mean(stats["loss_student"]), 4),
            np.round(np.mean(stats["loss_teacher"]), 4),

            np.round(np.mean(stats["ps"]), 4),
            np.round(np.mean(stats["rs"]), 4),
            np.round(np.mean(stats["fs"]), 4),
        )


    def adjust_learning_rate(self, epoch):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.alpha_plan[epoch]
            param_group["betas"] = (self.beta1_plan[epoch], 0.999)


    # ------------------------- Evaluation ------------------------- #
    def evaluate_model(self, model, test_loader, use_ema: bool = False, ema_smoother: EMASmoother = None):
        """
        model: 待评估模型
        use_ema: 是否使用 EMA 参数
        ema_smoother: 对应的 EMASmoother 对象
        """
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels, _ in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                if use_ema and ema_smoother is not None:
                    # 使用 EMA 参数进行前向推理
                    logits = ema_smoother.forward_with_shadow(model, images)
                else:
                    logits = model(images)
                preds = torch.argmax(logits, dim=1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        return 100.0 * correct / total

    def evaluate(self, test_loader):
        """
        返回：
            学生原模型准确率, 学生EMA准确率
        """
        acc_student = self.evaluate_model(self.student, test_loader, use_ema=False)
        acc_student_ema = self.evaluate_model(self.student, test_loader, use_ema=True, ema_smoother=self.student_ema)
        return acc_student, acc_student_ema
    
        
    def collect_and_draw(self, train_loader, epoch=200, save_dir="/home/shunjie/codes/tnnls_code/new/bss", sample_size=5000):
        self.student.train()
        self.teacher.train()

        all_embeddings = []
        all_indices = []

        pred_clean = []   # 学生模型判定的干净样本（全局索引）
        pred_noisy = []   # 学生模型判定的噪声样本（全局索引）

        for _, (images, labels, indices) in enumerate(train_loader):
            indices = indices.cpu().numpy()
            images, labels = images.to(self.device), labels.to(self.device)

            # Teacher filtering
            if epoch > self.early_stop:
                images_clean, labels_clean, _, _, _, _, idx_clean = check_given_samples(
                    self.teacher, images, labels, self.p_thresh, indices, self.noise_or_not, device=self.device
                )
            else:
                images_clean, labels_clean, idx_clean = images, labels, indices

            # Student vs Student EMA
            logits_student = self.student(images_clean)
            with torch.no_grad():
                logits_student_ema = self.student_ema.forward_with_shadow(self.student, images_clean)

            # Student loss
            loss_student, _, _, _, _, _, _, _, _, ind_update = self.loss_fn(
                logits_student,
                logits_student_ema,
                images_clean,
                labels_clean,
                self.s_thresh if epoch > self.early_stop else self.rate_schedule[epoch],
                idx_clean,
                self.noise_or_not,
                self.co_lambda,
            )

            ind_update = list(ind_update)

            # 获取 embedding
            _, embeddings = self.student(images, feature=True)
            embeddings = embeddings.detach().cpu()
            all_embeddings.append(embeddings)
            all_indices.append(indices)

            # 全局 clean / noisy
            clean_global = indices[ind_update]
            noisy_global = np.setdiff1d(indices, clean_global)

            pred_clean.extend(clean_global.tolist())
            pred_noisy.extend(noisy_global.tolist())

        # 拼接 batch
        all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
        all_indices = np.concatenate(all_indices, axis=0)

        # 随机采样
        if len(all_indices) > sample_size:
            sampled_idx = np.random.choice(len(all_indices), sample_size, replace=False)
            all_embeddings = all_embeddings[sampled_idx]
            all_indices = all_indices[sampled_idx]

            pred_clean = [idx for idx in pred_clean if idx in all_indices]
            pred_noisy = [idx for idx in pred_noisy if idx in all_indices]

        # t-SNE 降维
        tsne = TSNE(n_components=2, random_state=42, perplexity=50)
        emb_2d = tsne.fit_transform(all_embeddings)

        # 标记错误样本
        gt_clean = np.array([i for i in all_indices if self.noise_or_not[i]])
        missed_clean = np.setdiff1d(gt_clean, pred_clean)  # 实际干净但未被预测为干净
        gt_noisy = np.array([i for i in all_indices if not self.noise_or_not[i]])
        wrongly_selected_noisy = np.intersect1d(gt_noisy, pred_clean)  # 实际噪声但被预测为干净

        # 绘图
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))

        # 字体大小设置
        # axis_label_fontsize = 16
        title_fontsize = 20
        legend_fontsize = 16
        tick_fontsize = 16

        # 左图：Predicted Clean / Noisy
        if len(pred_clean) > 0:
            axs[0].scatter(
                emb_2d[np.isin(all_indices, pred_clean), 0],
                emb_2d[np.isin(all_indices, pred_clean), 1],
                c='#3274a1', s=10, label='PC'
            )
        if len(pred_noisy) > 0:
            axs[0].scatter(
                emb_2d[np.isin(all_indices, pred_noisy), 0],
                emb_2d[np.isin(all_indices, pred_noisy), 1],
                c='#e1812c', s=10, label='PN'
            )
        # 错误样本圈
        if len(missed_clean) > 0:
            axs[0].scatter(
                emb_2d[np.isin(all_indices, missed_clean), 0],
                emb_2d[np.isin(all_indices, missed_clean), 1],
                edgecolors='red', facecolors='none', s=15, linewidths=1.0, label='MC'
            )
        if len(wrongly_selected_noisy) > 0:
            axs[0].scatter(
                emb_2d[np.isin(all_indices, wrongly_selected_noisy), 0],
                emb_2d[np.isin(all_indices, wrongly_selected_noisy), 1],
                edgecolors='green', facecolors='none', s=15, linewidths=1.0, label='SN'
            )

        axs[0].legend(fontsize=legend_fontsize, loc='upper left')
        axs[0].set_title(f"Prediction", fontsize=title_fontsize)
        axs[0].tick_params(axis='x', labelsize=tick_fontsize)
        axs[0].tick_params(axis='y', labelsize=tick_fontsize)
        # axs[0].set_xlabel("X", fontsize=axis_label_fontsize)
        # axs[0].set_ylabel("Y", fontsize=axis_label_fontsize)

        # 右图：Ground Truth
        axs[1].scatter(
            emb_2d[np.isin(all_indices, gt_clean), 0],
            emb_2d[np.isin(all_indices, gt_clean), 1],
            c='#3274a1', s=10, label='C'
        )
        axs[1].scatter(
            emb_2d[np.isin(all_indices, gt_noisy), 0],
            emb_2d[np.isin(all_indices, gt_noisy), 1],
            c='#e1812c', s=10, label='N'
        )
        axs[1].legend(fontsize=legend_fontsize, loc='upper left')
        axs[1].set_title("Ground Truth", fontsize=title_fontsize)
        axs[1].tick_params(axis='x', labelsize=tick_fontsize)
        axs[1].tick_params(axis='y', labelsize=tick_fontsize)
        # axs[1].set_xlabel("X", fontsize=axis_label_fontsize)
        # axs[1].set_ylabel("Y", fontsize=axis_label_fontsize)

        plt.tight_layout()

        # 保存
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"dataset={self.args.dataset}_noise_rate={self.args.noise_rate}_noise_type={self.args.noise_type}_epoch={epoch}.png"), dpi=300)
        plt.savefig(os.path.join(save_dir, f"dataset={self.args.dataset}_noise_rate={self.args.noise_rate}_noise_type={self.args.noise_type}_epoch={epoch}.pdf"))
        plt.close(fig)
        print(f"[Visualization] Saved")
        
        
# Predicted Clean: PC
# Predicted Noisy: PN
# Missed Clean :MC
# Selected Noisy: SN
# Clean :C
# Noisy: N