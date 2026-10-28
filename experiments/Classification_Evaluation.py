import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

# File paths
predicted_cnn_metabolome_path = "../results/external_metabolome_predictions_CNN.csv"
test_original_microbiome_path = "../data/IBD/CLR_unfiltered_data/micro_test_clr.csv"
test_original_metabolome_path = "../data/IBD/CLR_unfiltered_data/metab_test_clr.csv"
predicted_MLP_metabolome_path = "../results/external_metabolome_predictions_MLP.csv"
test_metadata_path = "../data/IBD/CLR_unfiltered_data/external_metadata_IBD.csv"
predicted_pcmdm2_metabolome_path = "../results/external_metabolome_predictions_pcmdm2.csv"
# Load  data
test_original_microbiome = pd.read_csv(test_original_microbiome_path, index_col=0).values.T
test_original_metabolome = pd.read_csv(test_original_metabolome_path, index_col=0).values.T
predicted_MLP_metabolome = pd.read_csv(predicted_MLP_metabolome_path, index_col=None).values
predicted_cnn_metabolome = pd.read_csv(predicted_cnn_metabolome_path, index_col=None).values
predicted_pcmdm2_metabolome= pd.read_csv(predicted_pcmdm2_metabolome_path, index_col=None).values
test_metadata = pd.read_csv(test_metadata_path, index_col=0)
test_labels = test_metadata["Study.Group"].map(lambda x: 1 if x in ["UC", "CD"] else 0).values  # 1=疾病, 0=健康

# Dataset dictionary
datasets = {
    "Measured Microbiome": test_original_microbiome,
    "Measured Metabolome": test_original_metabolome,
    "MLP Predicted Metabolome": predicted_MLP_metabolome,
    "CNN Predicted Metabolome": predicted_cnn_metabolome,
    "pcmdm2 Predicted Metabolome": predicted_pcmdm2_metabolome
}

# Cross-validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=123)

# Store results
results_rf = []
results_lr = []

# Iterate over each dataset
for dataset_name, data in datasets.items():
    accuracies_rf, aucs_rf, precisions_rf, recalls_rf, f1s_rf = [], [], [], [], []
    accuracies_lr, aucs_lr, precisions_lr, recalls_lr, f1s_lr = [], [], [], [], []

    for train_index, test_index in kf.split(data):
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = test_labels[train_index], test_labels[test_index]

        # Train LASSO Logistic Regression (L1 regularization)
        clf_lr = LogisticRegression(penalty='l1', solver='liblinear', random_state=123)
        clf_lr.fit(X_train, y_train)

        # Predict (LASSO Logistic Regression)
        y_pred_lr = clf_lr.predict(X_test)
        y_prob_lr = clf_lr.predict_proba(X_test)[:, 1]

        # Compute classification metrics (LASSO Logistic Regression)
        accuracies_lr.append(accuracy_score(y_test, y_pred_lr))
        aucs_lr.append(roc_auc_score(y_test, y_prob_lr))
        precisions_lr.append(precision_score(y_test, y_pred_lr))
        recalls_lr.append(recall_score(y_test, y_pred_lr))
        f1s_lr.append(f1_score(y_test, y_pred_lr))


    # Compute 5-fold mean metrics (LASSO Logistic Regression)
    mean_acc_lr = np.mean(accuracies_lr)
    mean_auc_lr = np.mean(aucs_lr)
    mean_precision_lr = np.mean(precisions_lr)
    mean_recall_lr = np.mean(recalls_lr)
    mean_f1_lr = np.mean(f1s_lr)

    results_lr.append([dataset_name, mean_acc_lr, mean_auc_lr, mean_precision_lr, mean_recall_lr, mean_f1_lr])
    print(f"📌 {dataset_name} (LASSO Logistic Regression): Accuracy = {mean_acc_lr:.4f}, AUC = {mean_auc_lr:.4f}, Precision = {mean_precision_lr:.4f}, Recall = {mean_recall_lr:.4f}, F1-score = {mean_f1_lr:.4f}\n")

