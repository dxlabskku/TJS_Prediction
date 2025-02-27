
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

def plot_performance_with_auroc(
    model,
    X_test,
    y_test,
    device,
    threshold,
    train_loss_log,
    valid_loss_log
):

    model.eval()

    start_inference = time.time()
    with torch.no_grad():
        outputs_test = model(X_test.to(device))
        y_test_prob = (torch.sigmoid(outputs_test)).cpu().numpy()
        y_test_pred = (y_test_prob > threshold).astype(int)
    inference_time = time.time() - start_inference
    print(f"Inference Time: {inference_time:.2f} seconds")

    cm = confusion_matrix(y_test, y_test_pred)
    report = classification_report(y_test, y_test_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    fpr, tpr, _ = roc_curve(y_test, y_test_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 9))

    # Confusion Matrix
    plt.subplot(3, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix", fontsize=14)
    plt.ylabel("True Value", fontsize=14)
    plt.xlabel("Predicted Value", fontsize=14)

    # Classification Report
    print("\nClassification Report:")
    print(report_df)

    plt.subplot(3, 2, 2)
    sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap='Blues', vmin=0, vmax=1)
    plt.title("Precision, Recall, F1 Score Matrix", fontsize=14)

    # AUROC
    plt.subplot(3, 2, 3)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC={roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title("ROC-AUC", fontsize=14)
    plt.legend(loc="lower right")

    # Scatter: True vs. Predicted Probability
    plt.subplot(3, 2, 4)
    plt.scatter(y_test_prob, y_test, alpha=0.3)
    plt.xlabel("Predicted Probability", fontsize=14)
    plt.ylabel("True Value", fontsize=14)
    plt.title("True vs. Predicted", fontsize=14)

    # Training/Validation Loss
    plt.subplot(3, 2, 5)
    plt.plot(train_loss_log, label='Train Loss', color='blue')
    plt.plot(valid_loss_log, label='Valid Loss', color='orange')
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title('Train/Validation Loss', fontsize=14)
    plt.legend()

    plt.tight_layout()
    plt.show()
