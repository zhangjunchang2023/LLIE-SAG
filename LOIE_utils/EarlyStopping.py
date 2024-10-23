import numpy as np
import torch

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, save_path='best_model_guonihe3.pth'):
        """
        Args:
            patience (int): 当验证集上的性能不再改善时，允许继续训练的最大周期数
            min_delta (float): 性能改善的最小变化幅度
            save_path (str): 模型权重保存路径
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False
        self.save_path = save_path

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            # 保存当前最好的模型权重
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.save_path)
            print(f"Model improved. Saving model to {self.save_path}")
        else:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
