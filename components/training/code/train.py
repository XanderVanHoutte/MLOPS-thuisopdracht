import os
import argparse
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from pickle import dump
from azureml.core import Run

def classification_task(estimator, features, labels):
    """
    Evaluates classification by predicting ("predict") and evaluation ("score") of the modelling algorithm.
    """
    predictions = estimator.predict(features)
    run = Run.get_context()
    
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    
    # Log the metrics to Azure ML
    run.log("accuracy", accuracy)
    run.log("f1_score", f1)

    print(f"Accuracy: {accuracy}")
    print(f"F1 score: {f1}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_features", type=str, help="Path to training features")
    parser.add_argument("--training_labels", type=str, help="Path to training labels")
    parser.add_argument("--validation_features", type=str, help="Path to validation features")
    parser.add_argument("--validation_labels", type=str, help="Path to validation labels")
    parser.add_argument("--output_model", type=str, help="Path to save the trained model")
    parser.add_argument("--n_estimators", type=int, default=35, help="Number of estimators")
    parser.add_argument("--max_depth", type=int, default=15, help="Max depth of trees")
    parser.add_argument("--min_samples_leaf", type=int, default=4, help="Min samples per leaf")
    args = parser.parse_args()

    # Load data
    train_features = np.load(os.path.join(args.training_features, "train_features.npy"))
    train_labels = np.load(os.path.join(args.training_labels, "train_labels.npy"))
    val_features = np.load(os.path.join(args.validation_features, "val_features.npy"))
    val_labels = np.load(os.path.join(args.validation_labels, "val_labels.npy"))

    # Train the model
    model = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth, min_samples_leaf=args.min_samples_leaf)
    model.fit(train_features, train_labels)

    # Save the model
    with open(args.output_model, "wb") as f:
        dump(model, f)

    # Evaluate the model
    classification_task(model, train_features, train_labels)
    classification_task(model, val_features, val_labels)

if __name__ == "__main__":
    main()
