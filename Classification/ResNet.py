
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.models import resnet50, ResNet50_Weights
from cosmicannealing import CosineAnnealingWarmUpRestarts

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

class ResNet50Classifier(nn.Module):
    def __init__(self, threshold, pos_weight):
        super(ResNet50Classifier, self).__init__()
        # load pretrained resnet50
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        # Change the input channels of the first convolutional layer to 1 (assuming grayscale images)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Modify the output layer for the classification task
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1)
        self.relu = nn.ReLU(inplace=False)  # inplace=False
        self.device=device
        self.pos_weight=pos_weight
        self.BCEWithLogitsLoss = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(self.device)
        self.threshold=threshold
    def forward(self, x):
        x=x.unsqueeze(1)
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.resnet.fc(x)
        return x

    def train_model(self, train_loader, valid_loader, epochs, learning_rate, patience, l2_reg, l1_reg):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0, weight_decay=l2_reg)
        # self.scheduler = StepLR(self.optimizer, step_size=50, gamma=0.5)
        self.scheduler = CosineAnnealingWarmUpRestarts(self.optimizer, T_0=30, T_mult=1, eta_max=learning_rate,  T_up=10, gamma=0.9)

        best_loss = float('inf')
        epochs_no_improve = 0
        best_model_wts = None  # Save best model weights
        loss_log = []
        valid_log=[]
        for e in range(epochs):

            epoch_loss = 0
            self.train()
            for _, (data, target) in enumerate(train_loader):
                self.optimizer.zero_grad() #optimizer reset
                data, target = data.to(self.device), target.to(self.device, dtype=torch.float32)
                out = self.forward(data)
                # calculate loss
                loss = self.BCEWithLogitsLoss(out, target)
                l1_loss = 0
                for param in self.parameters():
                    l1_loss += torch.sum(torch.abs(param))

                # L1 regularization
                loss = loss + l1_reg * l1_loss
                epoch_loss += loss.item() * data.size(0)
                loss.backward()
                self.optimizer.step()
            # Scheduler
            self.scheduler.step()
            # Log current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            # Calculate average batch loss
            epoch_loss /= len(train_loader.dataset)
            loss_log.append(epoch_loss)

            # Evaluate model on validation set
            self.eval()
            valid_acc, valid_loss = self.predict(valid_loader)
            self.train()

            # Handle if valid_acc and valid_loss are arrays
            valid_acc = np.mean(valid_acc) if isinstance(valid_acc, np.ndarray) else valid_acc
            valid_loss = np.mean(valid_loss) if isinstance(valid_loss, np.ndarray) else valid_loss
            valid_log.append(valid_loss)

            print(f'>> [Epoch {e+1}/{epochs}] Total epoch loss: {epoch_loss:.4f} / Valid accuracy: {100*valid_acc:.1f}% / Valid loss: {valid_loss:.4f} / LR : {current_lr}')

            # Early Stopping
            if valid_loss < best_loss and (e>5):
                best_loss = valid_loss
                best_model_wts = self.state_dict()  # Save best model weights
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {e+1}")
                    self.load_state_dict(best_model_wts)
                    print(f'Best model loaded with validation loss:{best_loss}')
                    break

        # Load best model weights
        if best_model_wts is not None:
            self.load_state_dict(best_model_wts)
            print("Best model loaded with validation loss:", best_loss)

        return loss_log, best_model_wts, best_loss, valid_log

    def predict(self, valid_loader, return_preds=False):
        BCEWithLogitsLoss = nn.BCEWithLogitsLoss(reduction='sum')
        preds = []
        outs = []
        total_loss = 0
        correct = 0
        len_data = 0

        with torch.no_grad():
            for _, (data, target) in enumerate(valid_loader):
                data, target = data.to(self.device), target.to(self.device, dtype=torch.float32)
                # Model predictions
                out = self.forward(data)
                # Calculate loss
                loss = BCEWithLogitsLoss(out, target)
                total_loss += loss.item()  # Convert loss to scalar
                out_sigmoid = torch.sigmoid(out)
                outs += list(out_sigmoid.detach().cpu().numpy())
                pred = (out_sigmoid > self.threshold).float().detach().cpu().numpy()
                preds += list(pred)

                # Calculate accuracy
                correct += (pred == target.detach().cpu().numpy()).sum()
                len_data += target.size(0)

            acc = correct / len_data  # Calculate accuracy
            avg_loss = total_loss / len_data  # Calculate average loss

        if return_preds:
            return acc, avg_loss, preds, outs
        else:
            return acc, avg_loss

