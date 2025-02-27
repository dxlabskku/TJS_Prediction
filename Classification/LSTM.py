
import torch
import torch.nn as nn
import numpy as np
from cosmicannealing import CosineAnnealingWarmUpRestarts

class LSTMModel(nn.Module):
    def __init__(self, device, input_dim, hidden_dim, pos_weight, dropout, num_layers, threshold):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.threshold = threshold
        self.pos_weight = pos_weight
        self.dropout = dropout

        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            batch_first=True,
            bidirectional=True
        )
        self.ln = nn.LayerNorm(self.hidden_dim * 2)
        self.fc = nn.Linear(self.hidden_dim * 2, 1)
        self.BCEWithLogitsLoss = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(self.pos_weight)
        ).to(self.device)

    def forward(self, x):
        out, _ = self.lstm(x)
        # Layer Norm
        out = self.ln(out)
        # Mean pooling
        out = torch.mean(out, dim=1)
        out = self.fc(out)
        return out

    def train_model(self, train_loader, valid_loader, epochs, learning_rate, patience, l2_reg, l1_reg):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0, weight_decay=l2_reg)
        self.scheduler = CosineAnnealingWarmUpRestarts(
            self.optimizer,
            T_0=25,
            T_mult=1,
            eta_max=learning_rate,
            T_up=10,
            gamma=1.0
        )
        best_loss = float('inf')
        epochs_no_improve = 0
        best_model_wts = None

        train_loss_log = []
        valid_loss_log = []

        for epoch in range(epochs):
            self.train()
            epoch_loss = 0.0
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device, dtype=torch.float32)
                self.optimizer.zero_grad()
                out = self.forward(data)
                loss = self.BCEWithLogitsLoss(out, target)

                # L1
                l1_val = 0
                for param in self.parameters():
                    l1_val += torch.sum(torch.abs(param))
                loss += l1_reg * l1_val

                epoch_loss += loss.item() * data.size(0)
                loss.backward()
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
                self.optimizer.step()

            self.scheduler.step()
            epoch_loss /= len(train_loader.dataset)
            train_loss_log.append(epoch_loss)

            # Validation
            self.eval()
            valid_acc, valid_loss_val = self.predict(valid_loader)
            self.train()

            valid_loss_log.append(valid_loss_val)
            print(f"Epoch [{epoch+1}/{epochs}] - "
                  f"Train Loss: {epoch_loss:.4f} | "
                  f"Valid Loss: {valid_loss_val:.4f} | "
                  f"Valid Acc: {100*valid_acc:.2f}%")

            if valid_loss_val < best_loss and (epoch > 15):
                best_loss = valid_loss_val
                best_model_wts = self.state_dict()
                print("Best model improved.")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    if best_model_wts is not None:
                        self.load_state_dict(best_model_wts)
                    break

        if best_model_wts is not None:
            self.load_state_dict(best_model_wts)
            print(f"Best model loaded with validation loss: {best_loss:.4f}")
        return train_loss_log, best_model_wts, best_loss, valid_loss_log

    def predict(self, loader, return_preds=False):
        BCEWithLogitsLoss = nn.BCEWithLogitsLoss(reduction='sum')
        total_loss = 0.0
        correct = 0
        total_samples = 0
        preds_list = []

        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device, dtype=torch.float32)
                out = self.forward(data)
                loss = BCEWithLogitsLoss(out, target)
                total_loss += loss.item()

                out_sigmoid = torch.sigmoid(out)
                preds = (out_sigmoid > self.threshold).float()
                preds_list.extend(preds.cpu().numpy())

                correct += (preds.cpu().numpy() == target.cpu().numpy()).sum()
                total_samples += target.size(0)

        avg_loss = total_loss / total_samples
        accuracy = correct / total_samples
        if return_preds:
            return accuracy, avg_loss, preds_list
        return accuracy, avg_loss
