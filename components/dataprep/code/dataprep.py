import os
import argparse
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from glob import glob
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def process_file(file, output_data, training_features_output, testing_features_output, validation_features_output, training_labels_output, testing_labels_output, validation_labels_output):

    music = pd.read_csv(file)
    # Remove what we don't need
    music.drop([10000, 10001, 10002, 10003, 10004], inplace = True)
    music.reset_index(inplace = True)
    music = music.drop(["index", "instance_id", "track_name", "obtained_date"], axis = 1)
    music = music.drop(music[music["artist_name"] == "empty_field"].index)

    top_20_artists = music["artist_name"].value_counts()[:20].sort_values(ascending = True)
    fig, ax = plt.subplots()
    ax.barh(top_20_artists.index, top_20_artists)
    ax.set_label("Number of songs per artist")
    ax.set_title("Songs per artist")
    plt.savefig(os.path.join(output_data, "artists.png"))
    plt.close(fig)

    # A lot of artists_names exist, which would complicate the dataset
    music.drop("artist_name", axis = 1, inplace = True)

    # Visualise the Keys
    # Create the figure and axis
    fig, ax = plt.subplots()
    sns.countplot(x = "key", data = music, palette = "ocean", order = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"], ax=ax)
    ax.set_title(f"Counts in each key")
    plt.savefig(os.path.join(output_data, "keys.png"))
    plt.close(fig)

    fig, ax = plt.subplots()
    sns.countplot(x = "music_genre", data = music, palette = "ocean", ax=ax)
    ax.set_title(f"Counts in each music_genre")
    plt.savefig(os.path.join(output_data, "genres.png"))
    plt.close(fig)

    # Dropping some more because of data analytics
    music = music.drop(music[music["tempo"] == "?"].index)
    music["tempo"] = music["tempo"].astype("float")
    music["tempo"] = np.around(music["tempo"], decimals = 2)

    numeric_features = music.drop(["key", "music_genre", "mode"], axis = 1)

    key_encoder = LabelEncoder()
    music["key"] = key_encoder.fit_transform(music["key"])

    mode_encoder = LabelEncoder()
    music["mode"] = mode_encoder.fit_transform(music["mode"])

    music_features = music.drop("music_genre", axis = 1)
    music_labels = music["music_genre"]

    scaler = StandardScaler()
    music_features_scaled = scaler.fit_transform(music_features)

    tr_val_f, test_features, tr_val_l, test_labels = train_test_split(music_features_scaled, music_labels, test_size = 0.1, stratify = music_labels)
    train_features, val_features, train_labels, val_labels = train_test_split(tr_val_f, tr_val_l, test_size = len(test_labels), stratify = tr_val_l)

    np.save(os.path.join(training_features_output, "train_features.npy"), train_features)
    np.save(os.path.join(validation_features_output, "val_features.npy"), val_features)
    np.save(os.path.join(testing_features_output, "test_features.npy"), test_features)
    np.save(os.path.join(training_labels_output, "train_labels.npy"), train_labels)
    np.save(os.path.join(validation_labels_output, "val_labels.npy"), val_labels)
    np.save(os.path.join(testing_labels_output, "test_labels.npy"), test_labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="path to input data")
    parser.add_argument("--output_logs", type=str, help="path to output data")
    parser.add_argument("--training_features", type=str, help="path to output data")
    parser.add_argument("--testing_features", type=str, help="path to output data")
    parser.add_argument("--validation_features", type=str, help="path to output data")
    parser.add_argument("--training_labels", type=str, help="path to output data")
    parser.add_argument("--testing_labels", type=str, help="path to output data")
    parser.add_argument("--validation_labels", type=str, help="path to output data")
    args = parser.parse_args()

    output_dir = args.output_logs
    plots_dir = os.path.join(output_dir, "plots")
    
    training_features_output = os.path.join(args.training_features, "output")
    testing_features_output = os.path.join(args.testing_features, "output")
    validation_features_output = os.path.join(args.validation_features, "output")
    training_labels_output = os.path.join(args.training_labels, "output")
    testing_labels_output = os.path.join(args.testing_labels, "output")
    validation_labels_output = os.path.join(args.validation_labels, "output")

    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(training_features_output, exist_ok=True)
    os.makedirs(testing_features_output, exist_ok=True)
    os.makedirs(validation_features_output, exist_ok=True)
    os.makedirs(training_labels_output, exist_ok=True)
    os.makedirs(testing_labels_output, exist_ok=True)
    os.makedirs(validation_labels_output, exist_ok=True)

    process_file(args.data, plots_dir, training_features_output, testing_features_output, validation_features_output, training_labels_output, testing_labels_output, validation_labels_output)

if __name__ == "__main__":
    main()
