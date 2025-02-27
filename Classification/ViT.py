#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import timm
import numpy as np
from cosmicannealing import CosineAnnealingWarmUpRestarts

def expand_to_3_channels(x):
    """
    Expand a (B, 1, H, W) tensor to (B, 3, H, W) by copying channel information.
    """
    return x.expand(-1, 3, -1, -1)

class BinaryClassificationViTModel(nn.Module):
    """
    Vision Transformer model for binary classification.
    """
    def __init__(self, threshold=0.5, pos_weight=5, num_classes=1):
        super(BinaryClassificationViTModel, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.threshold = threshold
        self.pos_weight = pos_weight
        self.vit = timm.create_model(
            'vit_base_patch16_224',
            pretrained=True,
            num_classes=0  # remove default classifier
        )
        self.fc = nn.Linear(self.vit.num_features, num_classes)
        self.BCEWithLogitsLoss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight)).to(self.device)

    def forward(self, x, return_attention=False):
        # Reshape to (B, 1, 224, 224) and expand channels
        b, seq_len, feats = x.shape
        # In case x is (B, H, W), convert to (B, 1, H, W)
        x = x.view(b, 1, 224, 224)
        x = expand_to_3_channels(x)

        # Extract features
        x = self.vit.forward_features(x)
        if return_attention:
          attentions = []
          for blk in self.vit.blocks:
              # Extract Query, Key, Value
              qkv = blk.attn.qkv(x)  # (batch_size, 3 * embed_dim, num_patches + 1)
              # print(f"qkv shape: {qkv.shape}")

              # Reshape
              batch_size, num_patches, embed_dim = x.shape
              num_heads = 12
              head_dim = embed_dim // num_heads
              qkv = qkv.reshape(batch_size, num_patches, 3, num_heads, head_dim)
              # print(f"qkv shape after reshape: {qkv.shape}")

              # Q, K, V
              qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, num_heads, num_patches, head_dim)
              q, k, v = qkv[0], qkv[1], qkv[2]
              # print(f"q shape: {q.shape}, k shape: {k.shape}, v shape: {v.shape}")
              # Calculate attention score
              attn = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1) ** 0.5))  # Scaled dot-product attention
              attn = attn.softmax(dim=-1)
              # print(f"attn shape: {attn.shape}")

              attentions.append(attn)

          # attention tensor의 shape: (num_layers, batch_size, num_heads, num_patches, num_patches)
          attentions_tensor = torch.stack(attentions)  # (num_layers, batch_size, num_heads, num_patches, num_patches)
          mean_attention = attentions_tensor.mean(dim=0)  # AVG over all layer
          max_attention = attentions_tensor.max(dim=0).values  # Max over all layer

          return x, mean_attention, max_attention

        cls_token = x[:, 0, :]
        logits = self.fc(cls_token)
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
            T_0=20,
            T_mult=1,
            eta_max=learning_rate,
            T_up=5,
            gamma=0.8
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
                # L1 regularization
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
            valid_acc, valid_loss = self.predict(valid_loader)
            self.train()

            valid_loss_log.append(valid_loss)
            print(f"Epoch [{epoch+1}/{epochs}] - "
                  f"Train Loss: {epoch_loss:.4f} | "
                  f"Valid Loss: {valid_loss:.4f} | "
                  f"Valid Acc: {100*valid_acc:.2f}%")

            if valid_loss < best_loss and (epoch > 10):
                best_loss = valid_loss
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
                preds_list.extend(preds.detach().cpu().numpy())
                correct += (preds.cpu().numpy() == target.cpu().numpy()).sum()
                total_samples += target.size(0)

        avg_loss = total_loss / total_samples
        accuracy = correct / total_samples
        if return_preds:
            return accuracy, avg_loss, preds_list
        return accuracy, avg_loss


