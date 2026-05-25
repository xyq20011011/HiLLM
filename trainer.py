from Datasets.dataset import get_dataloader
import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import time
import random
import warnings
from Models.LLM_Decoder import LLM_Encoder

def random_mask_target_p(p, rate):
    if rate == "uniform":
        rate = random.uniform(0, 1)
    elif rate == "gaussian":
        rate = random.gauss(0.5, 0.1)
        min(max(rate, 0), 1)
    target_p = p.clone()
    batch_size, n = target_p.shape
    for i in range(batch_size):
        non_zero_indices = (target_p[i] != 0).nonzero(as_tuple=False).squeeze()

        try:
            if len(non_zero_indices) > 0:
                # 计算当前行需要设为 0 的数量
                num_to_zero = int(len(non_zero_indices) * rate)

                # 随机选择位置并设为 0
                indices_to_zero = torch.randperm(len(non_zero_indices))[:num_to_zero]
                target_p[i, non_zero_indices[indices_to_zero]] = 0
        except:
            print(1)

    return target_p


def rank_loss(pred_theta, llm_theta, margin=0.1, num_pairs=1024):
    """
    pred_theta: (B, D)
    llm_theta:  (B, D)
    """
    B, D = pred_theta.shape
    device = pred_theta.device

    # 1. 一次性采样 pair
    i = torch.randint(0, B, (num_pairs,), device=device)
    j = torch.randint(0, B, (num_pairs,), device=device)

    # 去掉 i == j
    mask = i != j
    i = i[mask]
    j = j[mask]

    if i.numel() == 0:
        return torch.tensor(0.0, device=device)

    # 2. 取对应向量
    pred_i = pred_theta[i]      # (P, D)
    pred_j = pred_theta[j]

    llm_i = llm_theta[i]
    llm_j = llm_theta[j]

    # 3. 差分
    pred_diff = pred_i - pred_j     # (P, D)
    llm_diff = llm_i - llm_j

    # 4. 排序符号
    sign = torch.sign(llm_diff)     # (P, D)

    valid = sign.abs() > 0          # (P, D)

    # 5. hinge loss
    loss_mat = torch.relu(
        margin - pred_diff * sign
    )                               # (P, D)

    # 6. mask 掉无效维度
    loss_mat = loss_mat * valid

    # 7. 每个 pair 做 mean
    valid_cnt = valid.sum(dim=1)    # (P,)

    # 防止除 0
    valid_cnt = torch.clamp(valid_cnt, min=1)

    pair_loss = loss_mat.sum(dim=1) / valid_cnt  # (P,)

    # 8. 全局平均
    return pair_loss.mean()

class ModelInfo:
    def __init__(self):
        self.current_epoch = 0
        self.losses = []
        self.val_aucs = []
        self.test_aucs = []
        self.val_accs = []
        self.test_accs = []
        self.val_rmse = []
        self.test_rmse = []

    def add(self, train_loss, val_auc, test_auc, val_acc=None, test_acc=None, val_rmse=None, test_rmse=None):
        self.current_epoch += 1
        self.losses.append(train_loss)
        self.val_aucs.append(val_auc)
        self.test_aucs.append(test_auc)

        self.val_accs.append(val_acc)
        self.test_accs.append(test_acc)

        self.val_rmse.append(val_rmse)
        self.test_rmse.append(test_rmse)

    def is_best(self):
        current_auc = self.val_aucs[-1]
        for auc in self.val_aucs[:-1]:
            if auc > current_auc:
                return False
        return True

    def best(self):
        best_auc = 0
        best_test = None
        best_val_acc = None
        best_test_acc = None
        best_val_rmse = None
        best_test_rmse = None
        for i, auc in enumerate(self.val_aucs):
            if auc > best_auc:
                best_auc = auc
                best_test = self.test_aucs[i]
                best_val_acc = self.val_accs[i]
                best_test_acc = self.test_accs[i]
                best_val_rmse = self.val_rmse[i]
                best_test_rmse = self.test_rmse[i]

        return best_auc, best_test, best_val_acc, best_test_acc, best_val_rmse, best_test_rmse

    def best_target(self):
        best_target_auc = 0
        best_target_acc = 0
        best_target_rmse = 0
        for i, auc in enumerate(self.test_aucs):
            if auc > best_target_auc:
                best_target_auc = auc
                best_target_acc = self.test_accs[i]
                best_target_rmse = self.test_rmse[i]
        return best_target_acc, best_target_auc, best_target_rmse

    def plot(self, title=None):
        epochs = range(self.current_epoch)
        plt.plot(epochs, self.losses, label='Training Loss', marker='o')
        plt.title(f'{title} Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        plt.plot(epochs, self.val_aucs, label='Validation AUC', marker='o', color='orange')
        plt.plot(epochs, self.test_aucs, label='Test AUC', marker='^', color='green')
        plt.title(f'{title} Validation and Test AUC')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.legend()
        plt.show()

    def best_epoch(self):
        best_auc = 0
        best_epoch = -1
        for i, auc in enumerate(self.val_aucs):
            if auc > best_auc:
                best_auc = auc
                best_epoch = i
        return best_epoch


class Trainer:
    def __init__(self, model_name, data_path="data"):
        self.constant = None
        self.Q_matrix = None
        self.test_loader = None
        self.val_loader = None
        self.train_loader = None
        self.dataset_name = None
        self.learning_rate = 5e-4
        self.weight_decay = 0
        self.verbose = True
        self.show_plot = False

        self.model_path = os.path.join("saved_model", model_name)
        self.model_name = model_name
        self.model = None
        self.model_info = None
        self.gpu = True

    def init_model(self, model_classname, **kwargs):
        self.print("Initializing...")
        if not os.path.exists(self.model_path):
            self.print(f"{self.model_path} created")
            os.makedirs(self.model_path)

        self.model = model_classname(self.constant, self.Q_matrix.cuda(), **kwargs)
        if self.gpu:
            self.model.cuda()
        self.model_info = ModelInfo()

    def remove(self):
        print("Are you sure to remove ", self.model_name, "?[yes/no]")
        if input() == "yes":
            if os.path.exists(self.model_path):
                # 获取文件夹中所有文件的列表
                file_list = os.listdir(self.model_path)
                # 遍历文件列表并删除每个文件
                for file_name in file_list:
                    file_path = os.path.join(self.model_path, file_name)
                    try:
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                    except Exception as e:
                        print(f"Error deleting {file_path}: {e}")
        else:
            exit()

    def exist_model(self):
        torch_model_pth = os.path.join(self.model_path, "model")
        return os.path.exists(torch_model_pth)

    def load_model(self):
        torch_model_pth = os.path.join(self.model_path, "model")
        assert os.path.exists(torch_model_pth), f"{torch_model_pth} does not exist"
        self.print(f"Model loaded from {self.model_path}")
        self.model = torch.load(torch_model_pth)
        if self.gpu:
            self.model.cuda()

        data_pth = os.path.join(self.model_path, "info.pkl")
        with open(data_pth, 'rb') as file:
            self.model_info = pickle.load(file)

    def save_model(self):
        assert self.model is not None
        self.print(f"Model saved to {self.model_path}")
        torch_model_pth = os.path.join(self.model_path, "model")
        torch.save(self.model, torch_model_pth)

        data_pth = os.path.join(self.model_path, "info.pkl")
        with open(data_pth, 'wb') as file:
            pickle.dump(self.model_info, file)

    def load_data(self, name="ASSIST", batch_size=32, fold=1):
        self.dataset_name = name
        self.print(f"Loading data...")
        self.train_loader, self.val_loader, self.test_loader, self.Q_matrix, self.constant = get_dataloader(name=name,
                                                                                                            batch_size=batch_size,
                                                                                                            fold=fold)

    def train(self, dataset_name, num_epoch=None, to_epoch=None, loss_function=None, mask_ratio=0.6, mute_result=False, margin=0.1, rank_loss_lam=1):
        assert self.model is not None
        assert self.train_loader is not None
        if to_epoch is not None:
            num_epoch = to_epoch - self.model_info.current_epoch

        if loss_function is None:
            def loss_func(selected_out, selected_label, theta, out, p_mask, q_matrix):
                return nn.BCELoss()(selected_out, selected_label)

            loss_function = loss_func
        optimizer_all = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)


        dataset_path = os.path.join("../Datasets", dataset_name, "LLM_dataset.pkl")

        def exists_in_dataset(target_p_matrix, dataset):
            for _, saved_p, _ in dataset:
                if torch.equal(saved_p, target_p_matrix):
                    return True
            return False

        LLM_dataset = []

        if os.path.exists(dataset_path):
            with open(dataset_path, "rb") as f:
                loaded = pickle.load(f)

            # 恢复到 GPU
            for t, l, m in loaded:
                LLM_dataset.append((t.cuda(), l.cuda(), m.cuda()))



        if not os.path.exists(dataset_path):
            LLM_encoder = LLM_Encoder(q_matrix=self.model.expanded_Q_matrix, final_tree=self.model.concept_tree,
                                      p_text_path=os.path.join("../Datasets", dataset_name,
                                                               "problem_id_text_map.json"))
            print("Generating LLM out")
            for batch in tqdm(self.train_loader):
                p_matrix, target_p_matrix, sid = batch
                p_mask = (target_p_matrix != 0)

                if self.gpu:
                    target_p_matrix = target_p_matrix.cuda()
                    sid = sid.cuda()

                if exists_in_dataset(target_p_matrix, LLM_dataset):
                    print("skip")
                    continue

                out, theta = self.model(target_p_matrix, sid)
                LLM_theta = LLM_encoder(target_p_matrix, theta, mask=p_mask)
                LLM_dataset.append((LLM_theta.detach(), target_p_matrix, p_mask))


                save_dataset = []
                for theta_saved, LLM_out, p_mask_saved in LLM_dataset:
                    save_dataset.append((
                        theta_saved.cpu(),
                        LLM_out.cpu(),
                        p_mask_saved.cpu()
                    ))

                with open(dataset_path, "wb") as f:
                    pickle.dump(save_dataset, f)


        for epoch in self.get_epoch_iter(
                range(self.model_info.current_epoch, self.model_info.current_epoch + num_epoch)):
            self.print(f"\nEpoch {epoch}")

            """
            Train on train set
            """
            self.model.train()
            self.print(f"Training...")
            sum_train_loss = 0
            time.sleep(0.1)

            """
            同步优化代码
            """
            for batch, (LLM_theta, LLM_p_matrix, LLM_p_mask) in self.get_batch_iter(zip(self.train_loader, LLM_dataset)):
                """
                Reconstruction Loss
                """
                p_matrix, target_p_matrix, sid = batch
                p_matrix = random_mask_target_p(target_p_matrix, mask_ratio)
                if self.gpu:
                    p_matrix, target_p_matrix, sid = p_matrix.cuda(), target_p_matrix.cuda(), sid.cuda()
                p_mask = (target_p_matrix != 0)

                out, theta = self.model(p_matrix, sid)
                selected_out = torch.masked_select(out, mask=p_mask)
                selected_label = torch.masked_select(target_p_matrix, mask=p_mask) - 1

                assert torch.max(selected_out) <= 1, f"{torch.max(selected_out)} > 1"
                assert torch.min(selected_out) >= 0, f"{torch.max(selected_out)} < 0"

                rec_loss = loss_function(selected_out, selected_label, theta, out, p_mask, self.Q_matrix.cuda())
                loss = rec_loss

                theta = self.model.assess_net(LLM_p_matrix)
                encoder_rank = rank_loss(
                    pred_theta=theta,
                    llm_theta=LLM_theta,
                    margin=margin
                )
                loss += rank_loss_lam * encoder_rank

                if not torch.isnan(loss):
                    optimizer_all.zero_grad()
                    loss.backward()
                    optimizer_all.step()
                    sum_train_loss += loss.item()


            train_loss = sum_train_loss / len(self.train_loader)
            time.sleep(0.1)
            self.print(f"Train Loss : {train_loss}\n")



            """
            Validate on val set
            """
            self.model.eval()
            self.print(f"Validating...")
            cat_label, cat_predict = [], []
            time.sleep(0.1)
            for batch in self.get_batch_iter(self.val_loader):
                p_matrix, target_p_matrix, sid = batch
                if self.gpu:
                    p_matrix, target_p_matrix, sid = p_matrix.cuda(), target_p_matrix.cuda(), sid.cuda()
                p_mask = (target_p_matrix != 0) & (p_matrix != target_p_matrix)

                out, theta = self.model(p_matrix, sid)
                selected_out = torch.masked_select(out, mask=p_mask)
                selected_label = torch.masked_select(target_p_matrix, mask=p_mask)

                cat_label += selected_label.unsqueeze(-1).detach().cpu().tolist()
                cat_predict += selected_out.detach().cpu().tolist()

            cat_label = np.array(cat_label) - 1
            cat_predict = np.array(cat_predict)
            val_auc = roc_auc_score(cat_label, cat_predict)
            val_acc = accuracy_score(cat_label, np.round(cat_predict))
            val_rmse = np.sqrt(mean_squared_error(cat_label, cat_predict))
            time.sleep(0.1)
            self.print(f"Validate AUC : {val_auc}\n")

            """
            Test on test set
            """
            self.print(f"Testing...")
            cat_label, cat_predict = [], []
            time.sleep(0.1)
            for batch in self.get_batch_iter(self.test_loader):
                p_matrix, target_p_matrix, sid = batch
                if self.gpu:
                    p_matrix, target_p_matrix, sid = p_matrix.cuda(), target_p_matrix.cuda(), sid.cuda()
                p_mask = (target_p_matrix != 0) & (p_matrix != target_p_matrix)

                out, theta = self.model(p_matrix, sid)
                selected_out = torch.masked_select(out, mask=p_mask)
                selected_label = torch.masked_select(target_p_matrix, mask=p_mask)

                cat_label += selected_label.unsqueeze(-1).detach().cpu().tolist()
                cat_predict += selected_out.detach().cpu().tolist()

            cat_label = np.array(cat_label) - 1
            cat_predict = np.array(cat_predict)
            test_auc = roc_auc_score(cat_label, cat_predict)
            test_acc = accuracy_score(cat_label, np.round(cat_predict))
            test_rmse = np.sqrt(mean_squared_error(cat_label, cat_predict))
            time.sleep(0.1)
            self.print(f"Test AUC : {val_auc}\n")

            self.model_info.add(train_loss, val_auc, test_auc, val_acc, test_acc, val_rmse, test_rmse)
            if self.model_info.is_best():
                self.save_model()
            if epoch % 10 == 0 and epoch != 0 and self.show_plot:
                self.plot()

        if not mute_result:
            print(f"""
Val best
Acc: {self.model_info.best()[2]}
Auc: {self.model_info.best()[0]}
Rmse: {self.model_info.best()[4]}
Test best 
Acc: {self.model_info.best()[3]}
Auc: {self.model_info.best()[1]}
Rmse: {self.model_info.best()[5]}""")


    def test(self, mask_ratio=0.2):
        """
        Test on test set
        """
        self.print(f"Testing...")
        cat_label, cat_predict = [], []
        time.sleep(0.1)
        response_num_arr = []
        for batch in self.get_batch_iter(self.test_loader):
            p_matrix, target_p_matrix, sid = batch
            if self.gpu:
                p_matrix, target_p_matrix, sid = p_matrix.cuda(), target_p_matrix.cuda(), sid.cuda()
            p_matrix = random_mask_target_p(target_p_matrix, mask_ratio)

            valid_flag = torch.sum(p_matrix, dim=1, keepdim=True) > 5  # shape: [batch_size, 1]
            valid_mask = valid_flag.expand(-1, p_matrix.size(1))  # shape: [batch_size, num_items]

            response_num = (torch.count_nonzero(p_matrix).item())/32
            response_num_arr.append(response_num)
            p_mask = (target_p_matrix != 0) & (p_matrix != target_p_matrix) & valid_mask

            out, theta = self.model(p_matrix, sid)
            selected_out = torch.masked_select(out, mask=p_mask)
            selected_label = torch.masked_select(target_p_matrix, mask=p_mask) - 1

            cat_label += selected_label.unsqueeze(-1).detach().cpu().tolist()
            cat_predict += selected_out.detach().cpu().tolist()

        avg_response_num = sum(response_num_arr) / len(response_num_arr)
        cat_label = np.array(cat_label)
        cat_predict = np.array(cat_predict)
        test_auc = roc_auc_score(cat_label, cat_predict)
        test_acc = accuracy_score(cat_label, np.round(cat_predict))
        test_rmse = np.sqrt(mean_squared_error(cat_label, cat_predict))
        self.print(f"Test auc: {test_auc}, mask_rate = {mask_ratio}, average response num = {avg_response_num}")
        return test_acc, test_auc, test_rmse

    def print(self, x):
        if self.verbose:
            print(x)

    def get_batch_iter(self, x):
        if self.verbose:
            return tqdm(x)
        else:
            return x

    def get_epoch_iter(self, x):
        if self.verbose:
            return x
        else:
            return tqdm(x)

