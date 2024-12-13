# components/validation/code/validate.py

import os
import argparse
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score, accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, roc_auc_score
from pickle import load
import matplotlib.pyplot as plt
import seaborn as sns



def main():
    """Main function of the script."""

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="path to input data")
    parser.add_argument("--testing_features", type=str, help="path to output data")
    parser.add_argument("--testing_labels", type=str, help="path to output data")
    parser.add_argument("--output_logs", type=str, help="path to output data")
    args = parser.parse_args()

    output_dir = args.output_logs
    plots_dir = os.path.join(output_dir, "plots")

    os.makedirs(plots_dir, exist_ok=True)

    testing_features = np.load(os.path.join(args.testing_features, "output", "test_features.npy"), allow_pickle=True)
    testing_labels = np.load(os.path.join(args.testing_labels, "output", "test_labels.npy"), allow_pickle=True)


    # LOAD IN THE AI MODEL THAT WAS SAVED
    with open(args.model, "rb") as f:
        model = load(f)

    print(classification_report(testing_labels, model.predict(testing_features)))

    plt.figure(figsize = (8, 6))
    sns.heatmap(confusion_matrix(testing_labels, model.predict(testing_features)),
        annot = True,
        fmt = ".0f",
        cmap = "vlag",
        linewidths = 2,
        linecolor = "red",
        xticklabels = model.classes_,
        yticklabels = model.classes_)
    plt.title("Actual values")
    plt.ylabel("Predicted values")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "heatmap.png"))


if __name__ == "__main__":
    main()