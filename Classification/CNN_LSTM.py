
import torch
import torch.nn as nn
import numpy as np
from cosmicannealing import CosineAnnealingWarmUpRestarts

class CNN_LSTM_Model(nn.Module):
    def __init__(self, device, input_dim, hidden_dim, num_layers, dropout, pos_weight, threshold):
        super(CNN_LSTM_Model, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pos_weight = pos_weight
        self.threshold = threshold
        self.BCEWithLogitsLoss = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(self.pos_weight)
        ).to(self.device)

        # CNN (1D)
        self.cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=input_dim,
                out_channels=128,
                kernel_size=6,
                stride=1,
                padding=3
            ),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Dropout(p=dropout),

            nn.Conv1d(
                in_channels=128,
                out_channels=256,
                kernel_size=6,
                stride=1,
                padding=3
            ),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Dropout(p=dropout)
        )

        # LSTM
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.ln = nn.LayerNorm(hidden_dim * 2)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        # Permute to (batch_size, input_dim, sequence_length) for CNN
        x = x.permute(0, 2, 1)
        out = self.cnn(x)
        # Now shape is (batch_size, 256, new_length)
        out = out.permute(0, 2, 1)
        # LSTM
        out, _ = self.lstm(out)
        out = self.ln(out)
        out = torch.mean(out, dim=1)  # Mean pooling over time
        out = self.fc(out)
        return out

    def train_model(self, train_loader, valid_loader, epochs, learning_rate, patience, l2_reg, l1_reg):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0, weight_decay=l2_reg)
        self.scheduler = CosineAnnealingWarmUpRestarts(
            self.optimizer,
            T_0=40,
            T_mult=1,
            eta_max=learning_rate,
            T_up=10,
            gamma=0.9
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

            if valid_loss_val < best_loss and (epoch > 10):
                best_loss = valid_loss_val
                best_model_wts = self.state_dict()
                epochs_no_improve = 0
                print("Best model improved.")
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
