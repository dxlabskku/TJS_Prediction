
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score
from cosmicannealing import CosineAnnealingWarmUpRestarts
import numpy as np


class CNN(nn.Module):
    def __init__(self, dropout, device, input_dim):
        super(CNN, self).__init__()
        self.dropout = dropout
        self.device = device
        self.input_dim = input_dim

        # Initial conv layer with 32 channels
        self.conv0 = nn.Conv1d(
            in_channels=1,
            out_channels=32,
            kernel_size=6,
            stride=1,
            padding=3
        )
        self.bn0 = nn.BatchNorm1d(32)
        self.pool0 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.conv1 = nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=6,
            stride=1,
            padding=3
        )
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv1d(
            in_channels=64,
            out_channels=128,
            kernel_size=6,
            stride=1,
            padding=3
        )
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.conv3 = nn.Conv1d(
            in_channels=128,
            out_channels=256,
            kernel_size=6,
            stride=1,
            padding=3
        )
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Calculate output size after convolutional layers
        conv_output_length = self.calculate_conv_output_length(input_dim)

        self.dropout_layer = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(256 * conv_output_length, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)

    def forward(self, x):
        # Squeeze the third dimension if it's size 1
        x = x.squeeze(2)

        x = self.pool0(F.relu(self.bn0(self.conv0(x))))
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout_layer(F.relu(self.fc1(x)))
        x = self.dropout_layer(F.relu(self.fc2(x)))
        x = self.dropout_layer(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x

    def calculate_conv_output_length(self, input_length):
        output_length = input_length
        # conv0 -> pool0
        output_length = (output_length + 2 * 3 - (6 - 1) - 1) // 2 + 1
        # conv1 -> pool1
        output_length = (output_length + 2 * 3 - (6 - 1) - 1) // 2 + 1
        # conv2 -> pool2
        output_length = (output_length + 2 * 3 - (6 - 1) - 1) // 2 + 1
        # conv3 -> pool3
        output_length = (output_length + 2 * 3 - (6 - 1) - 1) // 2 + 1

        return output_length

    def train_model(self, train_loader, valid_loader,
                    epochs, learning_rate, patience, l2_reg, l1_reg, i):
        self.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=0,
            weight_decay=l2_reg
        )

        self.scheduler = CosineAnnealingWarmUpRestarts(
            self.optimizer,
            T_0=30,
            T_mult=1,
            eta_max=learning_rate,
            T_up=10,
            gamma=0.9
        )
        best_loss = float('inf')
        epochs_no_improve = 0
        best_model_wts = None

        loss_log = []
        valid_loss_log = []

        criterion = nn.MSELoss()

        for e in range(epochs):
            epoch_loss = 0
            all_targets = []
            all_outputs = []

            self.train()
            for data, target in train_loader:
                self.optimizer.zero_grad()

                data = data.to(self.device)
                target = target.to(self.device, dtype=torch.float32)
                data = data.unsqueeze(1)
                target = target.unsqueeze(1)

                out = self.forward(data)
                loss = criterion(out, target)

                # L1 regularization
                l1_val = sum(torch.sum(torch.abs(param)) for param in self.parameters())
                loss = loss + l1_reg * l1_val

                epoch_loss += loss.item() * data.size(0)

                # For R² score
                all_targets.extend(target.cpu().numpy())
                all_outputs.extend(out.detach().cpu().numpy())

                loss.backward()
                self.optimizer.step()

            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']

            epoch_loss /= len(train_loader.dataset)
            train_r2 = r2_score(all_targets, all_outputs)
            loss_log.append(epoch_loss)

            valid_loss, valid_r2 = self.validate(valid_loader)
            valid_loss = np.mean(valid_loss) if isinstance(valid_loss, np.ndarray) else valid_loss
            valid_loss_log.append(valid_loss)

            print(f">> [Epoch {e+1}/{epochs}] "
                  f"Total epoch loss: {epoch_loss:.4f} / "
                  f"Train R²: {train_r2:.4f} / "
                  f"Validation loss: {valid_loss:.4f} / "
                  f"Validation R²: {valid_r2:.4f} / "
                  f"LR : {current_lr}")

            # Early stopping
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_model_wts = self.state_dict()
                epochs_no_improve = 0
                torch.save(
                    self.state_dict(),
                    f'.../1dcnn_regression_{i}.pth'
                )
                print("Best model saved as 'best_model'")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    best_model_wts = self.state_dict()
                    self.load_state_dict(best_model_wts)
                    print(f"Early stopping at epoch {e+1}")
                    break

        # Load best model weights
        if best_model_wts is not None:
            self.load_state_dict(best_model_wts)
            print("Best model loaded with validation loss:", best_loss)

        return loss_log, valid_loss_log

    def validate(self, valid_loader):
        criterion = nn.MSELoss()
        total_loss = 0.0
        all_targets = []
        all_outputs = []

        self.eval()
        with torch.no_grad():
            for data, target in valid_loader:
                data = data.to(self.device)
                target = target.to(self.device, dtype=torch.float32)
                data = data.unsqueeze(1)
                target = target.unsqueeze(1)

                out = self.forward(data)
                loss = criterion(out, target)
                total_loss += loss.item() * data.size(0)

                all_targets.extend(target.cpu().numpy())
                all_outputs.extend(out.detach().cpu().numpy())

        avg_loss = total_loss / len(valid_loader.dataset)
        avg_r2 = r2_score(all_targets, all_outputs)
        return avg_loss, avg_r2
