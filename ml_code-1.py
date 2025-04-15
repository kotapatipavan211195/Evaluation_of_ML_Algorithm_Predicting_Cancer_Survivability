# Description: This code evaluates the performance of different machine learning models on cancer survivability datasets.
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from sklearn.metrics import roc_auc_score, auc, roc_curve, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from typing import Tuple, Dict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths
path_in = os.getenv("DATA_PATH", "./data/")
path_out = os.getenv("OUTPUT_PATH", "./output/")
results = os.getenv("RESULT_PATH", "./results/")

# Function to load datasets dynamically
def load_datasets(path: str) -> Dict[str, pd.DataFrame]:
    """Load datasets from the specified path."""
    datasets = {}
    try:
        for filename in os.listdir(path):
            if filename.endswith(".csv"):
                datasets[filename.split('.')[0]] = pd.read_csv(os.path.join(path, filename))
        if not datasets:
            logging.warning("No datasets found in the specified path.")
    except Exception as e:
        logging.error(f"Error loading datasets: {e}")
    return datasets

# Load datasets
datasets = load_datasets(path_in)

# Define models
models = {
    "DT_gini": DecisionTreeClassifier(criterion="gini", min_samples_leaf=20),
    "DT_entropy": DecisionTreeClassifier(criterion="entropy", min_samples_leaf=10),
    "LR": LogisticRegression(max_iter=10000),
    "NB": GaussianNB(),
    "SVM": LinearSVC(dual=False, max_iter=10000),
    "RF_gini": RandomForestClassifier(criterion='gini', n_estimators=500, max_leaf_nodes=1000, n_jobs=-1),
    "RF_entropy": RandomForestClassifier(criterion='entropy', n_estimators=100, max_leaf_nodes=1000, n_jobs=-1),
    "MLP": MLPClassifier(solver='adam', hidden_layer_sizes=(10, 10), max_iter=10000),
    "ADA": AdaBoostClassifier(),
    "BGC": BaggingClassifier(),
    "GBC": GradientBoostingClassifier()
}

# Sampling techniques
sampling_techniques = {
    "No_Sampling": lambda X, Y: (X, Y),
    "Over_Sampling": lambda X, Y: RandomOverSampler().fit_resample(X, Y),
    "Under_Sampling": lambda X, Y: RandomUnderSampler().fit_resample(X, Y)
}

# Define stratified 10-fold cross-validation
skf = StratifiedKFold(n_splits=10, shuffle=False)

def get_data(data_key: str) -> Tuple[np.ndarray, np.ndarray]:
    """Retrieve input and output data from the datasets."""
    try:
        data = datasets[data_key].values
        return data[:, :-1], data[:, -1]
    except KeyError:
        logging.error(f"Dataset {data_key} not found.")
        return np.array([]), np.array([])

def train_and_evaluate_model(X: np.ndarray, Y: np.ndarray, model, sample: str, data_key: str, model_key: str):
    """Train and evaluate the model using stratified cross-validation."""
    nrl_confusion_matrix = []
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    i = 0

    for train_index, test_index in skf.split(X, Y):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        # Apply sampling
        X_train, Y_train = sampling_techniques[sample](X_train, Y_train)

        # Train the model
        try:
            model.fit(X_train, Y_train)
        except Exception as e:
            logging.error(f"Error training model {model_key}: {e}")
            return

        # Predict and evaluate
        Y_pred = model.predict(X_test)
        nrl_confusion_matrix.append(confusion_matrix(Y_test, Y_pred))
        fpr, tpr, _ = roc_curve(Y_test, Y_pred)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=2, alpha=0.3, label=f'ROC fold {i} (AUC = {roc_auc:.3f})')
        i += 1

    # Plot ROC curve
    plot_roc_curve(tprs, mean_fpr, aucs, data_key, model_key, sample)

    # Calculate metrics
    calculate_metrics(nrl_confusion_matrix)

def plot_roc_curve(tprs, mean_fpr, aucs, data_key: str, model_key: str, sample: str):
    """Plot and save the ROC curve."""
    mean_tpr = np.mean(tprs, axis=0)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black')
    plt.plot(mean_fpr, mean_tpr, color='blue',
             label=f'Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.4f})', lw=3, alpha=.8)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label='± 1 std. dev.')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.title(f'ROC Curve: {data_key} - {model_key} - {sample}')
    plt.savefig(os.path.join(results, f"{data_key}_{model_key}_{sample}_roc_curve.png"))
    plt.close()

def calculate_metrics(confusion_matrices):
    """Calculate and log evaluation metrics."""
    nrl_matrix_score = np.sum(confusion_matrices, axis=0)
    TN, FP, FN, TP = nrl_matrix_score.ravel()
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    f1_score = 2 * TP / (2 * TP + FP + FN)

    logging.info(f'Sensitivity: {sensitivity:.5f}')
    logging.info(f'Accuracy: {accuracy:.5f}')
    logging.info(f'Specificity: {specificity:.5f}')
    logging.info(f'Precision: {precision:.5f}')
    logging.info(f'F1_score: {f1_score:.5f}')

def main():
    """Main function to evaluate models on datasets."""
    for data_key in datasets.keys():
        X, Y = get_data(data_key)
        if X.size == 0 or Y.size == 0:
            continue
        for sample in sampling_techniques.keys():
            for model_key, model in models.items():
                logging.info(f"Evaluating {data_key} with {sample} and {model_key}")
                train_and_evaluate_model(X, Y, model, sample, data_key, model_key)

if __name__ == "__main__":
    main()
# End of the script
