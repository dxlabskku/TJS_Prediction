
import time
import random
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

%cd "..../Classification"
from Prepfortrain import data_split
from Prepfortrain import preprocess
from Visualization import plot_performance_with_auroc
from ResNet import ResNet50Classifier
from ViT import BinaryClassificationViTModel
from Transformer_Encoder import TimeSeriesTransformer
from CNN_LSTM import CNN_LSTM_Model
from LSTM import LSTMModel

###############################################################################
# Global device and random seed setup
###############################################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

###############################################################################
# Example: Suppose X, y are already loaded or created somewhere
# X, y = ...
# Make sure X.shape is (num_groups, sequence_length, num_features)
###############################################################################

###############################################################################
# 1. ResNet Training
###############################################################################
def train_resnet(X, y):
    print("========== Starting ResNet50 Training ==========")

    # Hyperparameters
    batch_size    = 16
    num_epochs    = 300
    learning_rate = 0.00006
    patience      = 20
    pos_weight    = torch.tensor(5)
    l2_reg        = 0
    l1_reg        = 0.000001
    threshold     = 0.5

    for i in range(100, 1001, 100):
        seed = i
        set_seed(seed)

        # Split data
        X_train_tensor, X_valid_tensor, X_test_tensor, y_train_tensor, y_valid_tensor, y_test_tensor = data_split(X, y, seed)

        # Build Dataset & DataLoader
        train_dataset = TensorDataset(X_train_tensor.to(device), y_train_tensor.to(device))
        valid_dataset = TensorDataset(X_valid_tensor.to(device), y_valid_tensor.to(device))
        test_dataset = TensorDataset(X_test_tensor.to(device), y_test_tensor.to(device))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

        # Initialize ResNet50 model
        model = ResNet50Classifier(threshold, pos_weight).to(device)

        print(f"Current seed: {seed}")
        if device.type == 'cuda':
            torch.cuda.reset_max_memory_allocated(device)

        start_time = time.time()
        loss_log, best_model_wts, best_loss, valid_log = model.train_model(
            train_loader,
            valid_loader,
            num_epochs,
            learning_rate,
            patience,
            l2_reg,
            l1_reg
        )
        total_train_time = time.time() - start_time
        print(f"Total Training Time: {total_train_time:.2f} seconds")

        if device.type == 'cuda':
            max_memory = torch.cuda.max_memory_allocated(device) / (1024**2)
            print(f"Maximum GPU memory allocated: {max_memory:.2f} MB")

        # Load best weights and evaluate
        model.load_state_dict(best_model_wts)
        plot_performance_with_auroc(
            model,
            X_test_tensor.cpu(),
            y_test_tensor.cpu().numpy(),
            device,
            threshold,
            loss_log,
            valid_log
        )

###############################################################################
# 2. Vision Transformer (ViT) Training
###############################################################################
def train_vit(X, y):
    print("========== Starting ViT Training ==========")

    import torch.nn.functional as F

    def pad_to_224(x):
        return F.pad(x, (0, 224 - x.shape[2], 0, 224 - x.shape[1]), "constant", -5)

    batch_size    = 16
    num_epochs    = 200
    learning_rate = 0.000004
    patience      = 20
    pos_weight    = torch.tensor(5)
    l2_reg        = 0.00003
    l1_reg        = 0
    threshold     = 0.5

    for i in range(100, 1001, 100):
        seed = i
        set_seed(seed)

        # Split
        X_train_tensor, X_valid_tensor, X_test_tensor, y_train_tensor, y_valid_tensor, y_test_tensor = data_split(X, y, seed)

        # Pad to 224x224
        X_train_tensor = pad_to_224(X_train_tensor)
        X_valid_tensor = pad_to_224(X_valid_tensor)
        X_test_tensor  = pad_to_224(X_test_tensor)

        # DataLoader
        train_dataset = TensorDataset(X_train_tensor.to(device), y_train_tensor.to(device))
        valid_dataset = TensorDataset(X_valid_tensor.to(device), y_valid_tensor.to(device))
        test_dataset = TensorDataset(X_test_tensor.to(device), y_test_tensor.to(device))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

        print(f"{i+1}th iteration | seed={seed}")
        model = BinaryClassificationViTModel(threshold, pos_weight).to(device)

        if device.type == 'cuda':
            torch.cuda.reset_max_memory_allocated(device)

        start_time = time.time()
        loss_log, best_model_wts, best_loss, valid_log = model.train_model(
            train_loader,
            valid_loader,
            num_epochs,
            learning_rate,
            patience,
            l2_reg,
            l1_reg
        )
        total_train_time = time.time() - start_time
        print(f"Total Training Time: {total_train_time:.2f} seconds")

        if device.type == 'cuda':
            max_memory = torch.cuda.max_memory_allocated(device) / (1024**2)
            print(f"Maximum GPU memory allocated: {max_memory:.2f} MB")

        model.load_state_dict(best_model_wts)
        plot_performance_with_auroc(
            model,
            X_test_tensor.cpu(),
            y_test_tensor.cpu().numpy(),
            device,
            threshold,
            loss_log,
            valid_log
        )

###############################################################################
# 3. Transformer Training
###############################################################################
def train_transformer(X, y):
    print("========== Starting Transformer Training ==========")

    batch_size    = 64
    num_epochs    = 300
    learning_rate = 0.000003
    patience      = 30
    pos_weight    = torch.tensor(5)
    l2_reg        = 0.0000001
    l1_reg        = 0.0000001
    threshold     = 0.5

    # Model config
    num_classes   = 1
    model_dim     = 512
    num_heads     = 16
    num_layers    = 12
    dropout       = 0

    for i in range(100, 1001, 100):
        seed = i
        set_seed(seed)

        X_train_tensor, X_valid_tensor, X_test_tensor, y_train_tensor, y_valid_tensor, y_test_tensor = data_split(X, y, seed)

        train_dataset = TensorDataset(X_train_tensor.to(device), y_train_tensor.to(device))
        valid_dataset = TensorDataset(X_valid_tensor.to(device), y_valid_tensor.to(device))
        test_dataset = TensorDataset(X_test_tensor.to(device), y_test_tensor.to(device))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

        print(f"{i+1}th iteration | seed={seed}")
        input_dim = X_train_tensor.shape[2]
        model = TimeSeriesTransformer(
            input_dim,
            model_dim,
            num_heads,
            num_layers,
            num_classes,
            dropout,
            device,
            pos_weight,
            threshold
        ).to(device)

        if device.type == 'cuda':
            torch.cuda.reset_max_memory_allocated(device)

        start_time = time.time()
        loss_log, best_model_wts, best_loss, valid_log = model.train_model(
            train_loader,
            valid_loader,
            num_epochs,
            learning_rate,
            patience,
            l2_reg,
            l1_reg
        )
        total_train_time = time.time() - start_time
        print(f"Total Training Time: {total_train_time:.2f} seconds")

        if device.type == 'cuda':
            max_memory = torch.cuda.max_memory_allocated(device) / (1024**2)
            print(f"Maximum GPU memory allocated: {max_memory:.2f} MB")

        plot_performance_with_auroc(
            model,
            X_test_tensor.cpu(),
            y_test_tensor.cpu().numpy(),
            device,
            threshold,
            loss_log,
            valid_log
        )

###############################################################################
# 4. CNN + LSTM Training
###############################################################################
def train_cnn_lstm(X, y):
    print("========== Starting CNN + LSTM Training ==========")
    from sklearn.metrics import classification_report

    batch_size   = 64
    hidden_dim   = 512
    num_epochs   = 300
    learning_rate= 0.000002
    patience     = 30
    pos_weight   = torch.tensor(5)
    l2_reg       = 0
    l1_reg       = 0
    dropout      = 0
    num_layers   = 4
    threshold    = 0.5

    for i in range(100, 1001, 100):
        seed = i
        set_seed(seed)

        X_train_tensor, X_valid_tensor, X_test_tensor, y_train_tensor, y_valid_tensor, y_test_tensor = data_split(X, y, seed)

        train_dataset = TensorDataset(X_train_tensor.to(device), y_train_tensor.to(device))
        valid_dataset = TensorDataset(X_valid_tensor.to(device), y_valid_tensor.to(device))
        test_dataset = TensorDataset(X_test_tensor.to(device), y_test_tensor.to(device))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

        print(f"{i+1}th iteration | seed={seed}")
        input_dim = X_train_tensor.shape[2]
        model = CNN_LSTM_Model(
            device,
            input_dim,
            hidden_dim,
            num_layers,
            dropout,
            pos_weight,
            threshold
        ).to(device)

        if device.type == 'cuda':
            torch.cuda.reset_max_memory_allocated(device)

        start_time = time.time()
        loss_log, best_model_wts, best_loss, valid_log = model.train_model(
            train_loader,
            valid_loader,
            num_epochs,
            learning_rate,
            patience,
            l2_reg,
            l1_reg
        )
        total_train_time = time.time() - start_time
        print(f"Total Training Time: {total_train_time:.2f} seconds")

        if device.type == 'cuda':
            max_memory = torch.cuda.max_memory_allocated(device) / (1024**2)
            print(f"Maximum GPU memory allocated: {max_memory:.2f} MB")

        plot_performance_with_auroc(
            model,
            X_test_tensor.cpu(),
            y_test_tensor.cpu().numpy(),
            device,
            threshold,
            loss_log,
            valid_log
        )

###############################################################################
# 5. LSTM Training
###############################################################################
def train_lstm(X, y):
    print("========== Starting LSTM Training ==========")
    from sklearn.metrics import classification_report

    batch_size   = 128
    hidden_dim   = 512
    num_epochs   = 500
    learning_rate= 0.000008
    patience     = 50
    pos_weight   = torch.tensor(5)
    l2_reg       = 0.00001
    l1_reg       = 0
    dropout      = 0.2
    num_layers   = 4
    threshold    = 0.5

    for i in range(100, 1001, 100):
        seed = i
        set_seed(seed)

        X_train_tensor, X_valid_tensor, X_test_tensor, y_train_tensor, y_valid_tensor, y_test_tensor = data_split(X, y, seed)

        train_dataset = TensorDataset(X_train_tensor.to(device), y_train_tensor.to(device))
        valid_dataset = TensorDataset(X_valid_tensor.to(device), y_valid_tensor.to(device))
        test_dataset = TensorDataset(X_test_tensor.to(device), y_test_tensor.to(device))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

        print(f"{i+1}th iteration | seed={seed}")
        input_dim = X_train_tensor.shape[2]
        model = LSTMModel(
            device,
            input_dim,
            hidden_dim,
            pos_weight,
            dropout,
            num_layers,
            threshold
        ).to(device)

        if device.type == 'cuda':
            torch.cuda.reset_max_memory_allocated(device)

        start_time = time.time()
        loss_log, best_model_wts, best_loss, valid_log = model.train_model(
            train_loader,
            valid_loader,
            num_epochs,
            learning_rate,
            patience,
            l2_reg,
            l1_reg
        )
        total_train_time = time.time() - start_time
        print(f"Total Training Time: {total_train_time:.2f} seconds")

        if device.type == 'cuda':
            max_memory = torch.cuda.max_memory_allocated(device) / (1024**2)
            print(f"Maximum GPU memory allocated: {max_memory:.2f} MB")

        plot_performance_with_auroc(
            model,
            X_test_tensor.cpu(),
            y_test_tensor.cpu().numpy(),
            device,
            threshold,
            loss_log,
            valid_log
        )

###############################################################################
# Main routine to sequentially train all models
###############################################################################
def main():
    """
    Main routine to train 5 different models in sequence:
      1) ResNet
      2) ViT
      3) Transformer
      4) CNN+LSTM
      5) LSTM
    """
    print("=== Starting Full Training Pipeline ===")
    X, y = preprocess()
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    # Uncomment these lines once X and y are ready:
    # train_resnet(X, y)
    # train_vit(X, y)
    # train_transformer(X, y)
    train_cnn_lstm(X, y)
    train_lstm(X, y)

    print("=== All Trainings Completed ===")

if __name__ == "__main__":
    main()
