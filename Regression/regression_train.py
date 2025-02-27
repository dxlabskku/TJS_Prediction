
import torch
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gc
import random
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
%cd "..../Regression"
from prepforreg import data_split
from prepforreg import preprocess
from regression_cnn import CNN
from visualization import plot_regression_performance

# --------------------------------------------------------------------------
# Global device and random seed setup
# --------------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --------------------------------------------------------------------------
# Main training routine
# --------------------------------------------------------------------------
def main(X, y):
    batch_size = 16
    num_epochs = 200
    learning_rate = 0.001
    patience = 15
    l2_reg = 0
    l1_reg = 0
    dropout = 0

    # DataFrame to accumulate SHAP top-10 feature information
    all_top_10_features = pd.DataFrame()

    # Iterate over multiple random seeds
    for i in range(102, 1003, 100):
        seed = i
        set_seed(seed)

        # Split data
        X_train, X_valid, X_test, y_train, y_valid, y_test = data_split(X, y, seed)

        # Create Datasets
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32).to(device),
            torch.tensor(y_train.values, dtype=torch.float32).to(device)
        )
        valid_dataset = TensorDataset(
            torch.tensor(X_valid, dtype=torch.float32).to(device),
            torch.tensor(y_valid.values, dtype=torch.float32).to(device)
        )
        test_dataset = TensorDataset(
            torch.tensor(X_test, dtype=torch.float32).to(device),
            torch.tensor(y_test.values, dtype=torch.float32).to(device)
        )

        # Create DataLoaders
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)
        test_loader  = DataLoader(dataset=test_dataset,  batch_size=batch_size, shuffle=False)

        input_dim = X_train.shape[2]  # Adjust if your shape differs
        print(f'{i}th iteration and random seed is: {random}\n'
              f'l1_reg : {l1_reg}\nl2_reg : {l2_reg}\n'
              f'batch_size : {batch_size}\nlearning_rate : {learning_rate}\ndropout : {dropout}')

        # Instantiate and train the model
        model = CNN(dropout, device, input_dim).to(device)
        train_loss_log, valid_loss_log = model.train_model(
            train_loader, valid_loader,
            num_epochs, learning_rate, patience,
            l2_reg, l1_reg, i
        )

        # Load best saved model
        best_model = CNN(dropout, device, input_dim).to(device)
        best_model.load_state_dict(torch.load(
            f'..../1dcnn_regression_{i}.pth'
        ))

        # Evaluate
        r2, rmse = plot_regression_performance(best_model, test_loader, device, train_loss_log, valid_loss_log)

        # SHAP Analysis
        best_model.eval()
        X_test_sampled = X_test
        X_test_tensor = torch.tensor(X_test_sampled, dtype=torch.float32).to(device)

        explainer = shap.GradientExplainer(best_model, X_test_tensor)
        shap_values_list = []
        for idx in tqdm(range(X_test_tensor.shape[0])):
            single_val = explainer.shap_values(X_test_tensor[idx].unsqueeze(0))
            shap_values_list.append(single_val[0])

        # Combine SHAP values into a single array
        shap_values = np.concatenate(shap_values_list, axis=0)

        # Compute top-10 feature importances
        columns = X.columns.tolist()
        shap_importances = np.abs(shap_values).mean(axis=0).flatten()
        top_10_indices_shap = np.argsort(shap_importances)[-10:][::-1]
        top_10_importances_shap = shap_importances[top_10_indices_shap]
        top_10_features_shap = [columns[idx] for idx in top_10_indices_shap]

        # Plot top-10 feature importances
        plt.figure(figsize=(4, 3))
        bars = plt.barh(range(len(top_10_features_shap)), top_10_importances_shap, align='center')
        plt.yticks(range(len(top_10_features_shap)), top_10_features_shap)
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Top 10 Feature Importances using SHAP')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        for bar in bars:
            plt.gca().text(
                bar.get_width(),
                bar.get_y() + bar.get_height()/2,
                f'{bar.get_width():.5f}',
                va='center', ha='left'
            )
        plt.show()

        # Reshape SHAP values for dependence plots
        shap_values_reshaped = shap_values.squeeze().reshape(X_test.shape[0], X_test.shape[2])
        X_test_sampled_reshaped = X_test_sampled.squeeze().reshape(X_test.shape[0], X_test.shape[2])

        # Store top-10 features
        top_10_df = pd.DataFrame({
            'r^2': [r2]*len(top_10_features_shap),
            'RMSE': [rmse]*len(top_10_features_shap),
            'Feature': top_10_features_shap,
            'Importance': top_10_importances_shap,
            'Iteration': [i]*len(top_10_features_shap)
        })
        all_top_10_features = pd.concat([all_top_10_features, top_10_df], ignore_index=True)

        print(f"Iteration {i} processed.")

        # 2x5 subplots for SHAP dependence plots
        fig, axes = plt.subplots(2, 5, figsize=(24, 8))
        for idx, feature in enumerate(top_10_features_shap):
            row, col = divmod(idx, 5)
            shap.dependence_plot(
                feature,
                shap_values_reshaped,
                X_test_sampled_reshaped,
                feature_names=columns,
                ax=axes[row, col],
                show=False,
                interaction_index=None
            )
            axes[row, col].set_title(f'{feature}', fontsize=14)
            axes[row, col].grid(True)
            axes[row, col].set_xlim([0, 1])    # Adjust if needed
            axes[row, col].set_ylim([-140, 80]) # Adjust if needed
            axes[row, col].set_xlabel(f'Scaled_{feature}', fontsize=14)
            axes[row, col].set_ylabel('SHAP_Value', fontsize=14)
            axes[row, col].tick_params(axis='x', labelsize=12)
            axes[row, col].tick_params(axis='y', labelsize=12)
            axes[row, col].grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
        plt.tight_layout()
        plt.show()

    # Save all top-10 features to CSV
    all_top_10_features.to_csv(
        f".../all_top10_features.csv",
        index=False
    )
    print("All top-10 SHAP features saved.")

if __name__ == "__main__":

    X, y = preprocess()
    main(X, y)

