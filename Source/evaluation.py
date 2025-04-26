from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def modelEvaluation(y_pred, y_pred_proba, y_test, labels, dataset):
    """
    Evaluate the model performance using various metrics.
    """
    cm = confusion_matrix(y_test, y_pred)

    # Accuracy score
    acc_score = accuracy_score(y_test, y_pred)
    # Precision score
    pre_score = precision_score(y_test, y_pred, average='macro')
    # Recall score
    rec_score = recall_score(y_test, y_pred, average='macro')
    # F1-score
    f1 = f1_score(y_test, y_pred, average='macro')
    # AUC score
    auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')

    # Draw metrics
    metrics_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
    metrics_values = [acc_score, pre_score, rec_score, f1, auc]
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        x=metrics_labels,
        y=metrics_values,
        hue=metrics_labels,  # Assign `hue` to `x` variable
        dodge=False,
        palette=["#FF6F61", "#92A8D1", "#88B04B", "#F7CAC9", "#61ffbd"],  # Add 5 colors
        legend=False  # Disable legend
    )
    for i, v in enumerate(metrics_values):
        ax.text(i, v - 0.04, f"{v:.4f}", ha='center', va='bottom', fontsize=10)
    plt.title("Model Performance Metrics")
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.xlabel("Metrics")
    plt.savefig(f"./benchmarks/{dataset}/model_performance.png")
    plt.close()


    # Draw confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, fmt='d', cmap='viridis', xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion Matrix", pad=20)  # Adjust title position
    plt.xlabel("Predicted", labelpad=15)  # Adjust x-axis label position
    plt.ylabel("Actual", labelpad=15)  # Adjust y-axis label position
    plt.savefig(f"./benchmarks/{dataset}/confusion_matrix.png", bbox_inches='tight')  # Ensure the image is tightly cropped

    return acc_score, pre_score, rec_score, f1, auc
