import torch
from abc import ABC
import torch.nn as nn
import torch.optim as optim
from CCLE.experiment.myutils import cross_entropy_loss

class Optimizer(nn.Module, ABC):
    def __init__(self, model, train_data, test_data, test_mask, train_mask, evaluate_fun,
                 lr=0.001, epochs=500, test_freq=20, device="cpu", alpha=0.9, beta=0.1):
        super(Optimizer, self).__init__()
        self.model = model.to(device)

        self.train_data = train_data.to(device)
        self.test_data = test_data.to(device)
        self.test_mask = test_mask.to(device).bool()
        self.train_mask = train_mask.to(device).bool()

        self.evaluate_fun = evaluate_fun
        self.lr = lr
        self.epochs = epochs
        self.test_freq = test_freq
        self.alpha = alpha
        self.beta = beta

        self.device = device

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0)

    def forward(self):
        true_data = torch.masked_select(self.test_data, self.test_mask)

        best_auc = -1
        best_epoch = -1
        best_predict = None

        for epoch in range(self.epochs):

            predict_data, contrastive_loss = self.model()

            loss_supervised = cross_entropy_loss(self.train_data, predict_data, self.train_mask)

            total_loss = self.alpha * loss_supervised + self.beta * contrastive_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            predict_data_masked = torch.masked_select(predict_data, self.test_mask)
            auc = self.evaluate_fun(true_data, predict_data_masked)

            if auc > best_auc:
                best_auc = auc
                best_epoch = epoch
                best_predict = predict_data_masked.clone().detach()

            if epoch % self.test_freq == 0 or epoch == self.epochs - 1:
                print(
                    f"epoch:{epoch:4d} "
                    f"loss: {total_loss.item():.6f} "
                    f"(supervised: {loss_supervised.item():.6f}, contrastive: {contrastive_loss.item():.6f}) "
                    f"auc: {auc:.4f} "
                    f"best_auc: {best_auc:.4f} at epoch {best_epoch}"
                )

        print("Training finished. Best AUC:", best_auc, "at epoch", best_epoch)

        return best_epoch, true_data, best_predict
