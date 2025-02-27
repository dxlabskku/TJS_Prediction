
import torch
import torch.nn as nn
import numpy as np
from cosmicannealing import CosineAnnealingWarmUpRestarts

class PositionalEncoding(nn.Module):
    """
    Adds sinusoidal positional encoding to the input embeddings.
    """
    def __init__(self, model_dim, max_len=224):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, model_dim, 2).float()
            * (-np.log(10000.0) / model_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, time, model_dim)
        x = x + self.pe[:, :x.size(1), :]
        return x

class TimeSeriesTransformer(nn.Module):
    """
    Transformer-based model for time-series classification.
    """
    def __init__(
        self,
        input_dim,
        model_dim,
        num_heads,
        num_layers,
        num_classes,
        dropout,
        device,
        pos_weight,
        threshold
    ):
        super(TimeSeriesTransformer, self).__init__()
        self.device = device
        self.threshold = threshold
        self.BCEWithLogitsLoss = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weight)
        ).to(self.device)

        # Embedding layer
        self.embedding = nn.Linear(input_dim, model_dim)
        # Positional encoding
        self.positional_encoding = PositionalEncoding(model_dim)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output FC
        self.fc = nn.Linear(model_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        x = self.embedding(x)
        x = self.transformer_encoder(x)  # (B, T, model_dim)
        x = x[:, -1, :]  # take last time step
        x = self.dropout(x)
        logits = self.fc(x)
        return logits

    def train_model(
        self,
        train_loader,
        valid_loader,
        epochs,
        learning_rate,
        patience,
        l2_reg,
        l1_reg
    ):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0, weight_decay=l2_reg)
        self.scheduler = CosineAnnealingWarmUpRestarts(
            self.optimizer,
            T_0=50,
            T_mult=1,
            eta_max=learning_rate,
            T_up=5,
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
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early Stopping at Epoch {epoch+1}")
                    break

        if best_model_wts is not None:
            self.load_state_dict(best_model_wts)
            print(f"Loaded best model with valid loss: {best_loss:.4f}")

        return train_loss_log, best_model_wts, best_loss, valid_loss_log

    def predict(self, data_loader, return_preds=False):
        BCEWithLogitsLoss = nn.BCEWithLogitsLoss(reduction='sum')
        total_loss = 0.0
        correct = 0
        total_samples = 0
        preds_list = []

        with torch.no_grad():
            for data, target in data_loader:
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
