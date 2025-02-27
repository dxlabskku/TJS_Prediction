
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.ticker as ticker
def thousands_formatter(x, pos):
    return f'{int(x/1000)}k'


def plot_regression_performance(model, test_loader, device, train_loss_log, valid_loss_log):
    model.eval()
    all_y_true, all_y_pred = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs_test = model(X_batch)
            all_y_true.append(y_batch.cpu().numpy())
            all_y_pred.append(outputs_test.cpu().numpy())

    # Flatten and concatenate all batches
    y_test = np.concatenate(all_y_true, axis=0).flatten()
    y_test_pred = np.concatenate(all_y_pred, axis=0).flatten()

    # Residuals, MAE, MSE, R²
    residuals = y_test - y_test_pred
    mae = mean_absolute_error(y_test, y_test_pred)
    mse = mean_squared_error(y_test, y_test_pred)
    r2 = r2_score(y_test, y_test_pred)

    # Plotting
    plt.figure(figsize=(11, 10))

    # 1. True vs Predicted Scatter Plot
    plt.subplot(3, 3, 1)
    plt.scatter(y_test, y_test_pred, alpha=0.6,s=15)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Perfect Prediction')
    z = np.polyfit(y_test, y_test_pred, 1)
    p = np.poly1d(z)
    plt.plot(y_test, p(y_test), color='blue', linestyle='-', linewidth=1, label='Trend Line')
    plt.grid(True)
    plt.xlabel('True Values (Day)', fontsize=11)
    plt.ylabel('Predicted Values (Day)', fontsize=11)
    plt.title(f'Days until Injury\nR^2 : {r2:.2f}', fontsize=11)
    plt.legend()

    bins = [0, 100, 200, 300, 400, 500]
    bin_indices = np.digitize(y_test, bins)

    rmses = []
    bin_labels = []
    counts = []  

    for i in range(1, len(bins)):
        in_bin = (bin_indices == i)
        count = np.sum(in_bin)  
        if count > 0:
            low = bins[i-1]
            high = bins[i]
            true_bin = y_test[in_bin]
            pred_bin = y_test_pred[in_bin]

            # RMSE 계산
            rmse = np.sqrt(np.mean((pred_bin - true_bin)**2))
        else:
            rmse = 0

        rmses.append(rmse)
        bin_labels.append(f"~{bins[i]}")
        counts.append(count)  

    plt.subplot(3, 3, 2)
    x_positions = np.arange(len(rmses))
    plt.bar(x_positions, rmses, color='skyblue', alpha=0.8)
    plt.xticks(x_positions, bin_labels, rotation=0, fontsize=11)
    plt.xlabel('Time-to-Injury Interval (Days)', fontsize=11)
    plt.ylabel('RMSE', fontsize=11)
    plt.ylim(0,140)
    plt.title(f'Interval RMSE\n0~100 : {rmses[0]:.2f}', fontsize=11)
    plt.grid(axis='y')

    for i, val in enumerate(rmses):
        count = counts[i]  
        offset = val * 0.02 if val > 0 else 0.5
        plt.text(i, val + offset, f"{val:.2f}\nn: {count}", ha='center', va='bottom', fontsize=9)

    plt.suptitle(f'R²: {r2:.4f}', fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Visualization of Train/Validation Loss Chart
    plt.subplot(3, 3, 3)
    plt.plot(train_loss_log, label='Train Loss', color='blue')
    plt.plot(valid_loss_log, label='Validation Loss', color='orange')
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title('Train/Validation Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(thousands_formatter))
    plt.show()


    return r2, rmses[0]
