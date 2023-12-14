import numpy as np
import pandas as pd


SEED = 123  # for reproducibility
np.random.seed(seed=SEED)

def make_data(sample_size = 300):
    """ Produce a 3-class classification dataset with 2-dimensional features and multiple noisy annotations per example. """
    num_annotators=50  # total number of data annotators
    class_frequencies = [0.5, 0.25, 0.25]
    sizes=[int(np.ceil(freq*sample_size)) for freq in class_frequencies]  # number of examples belonging to each class
    good_annotator_quality = 0.6
    bad_annotator_quality = 0.3
    
    # Underlying statistics of the datset (unknown to you)
    means=[[3, 2], [7, 7], [0, 8]]
    covs=[[[5, -1.5], [-1.5, 1]], [[1, 0.5], [0.5, 4]], [[5, 1], [1, 5]]]
    
    m = len(means)  # number of classes
    n = sum(sizes)
    local_data = []
    labels = []

    # Generate features and true labels
    for idx in range(m):
        local_data.append(
            np.random.multivariate_normal(mean=means[idx], cov=covs[idx], size=sizes[idx])
        )
        labels.append(np.array([idx for i in range(sizes[idx])]))
    X_train = np.vstack(local_data)
    true_labels_train = np.hstack(labels)

    # Generate noisy labels from each annotator
    s = pd.DataFrame(
        np.vstack(
            [
                generate_noisy_labels(true_labels_train, good_annotator_quality)
                if i < num_annotators - 10  # last 10 annotators are worse
                else generate_noisy_labels(true_labels_train, bad_annotator_quality)
                for i in range(num_annotators)
            ]
        ).transpose()
    )

    # Each annotator only labels approximately 10% of the dataset (unlabeled points represented with NaN)
    s = s.apply(lambda x: x.mask(np.random.random(n) < 0.9)).astype("Int64")
    s.dropna(axis=1, how="all", inplace=True)
    s.columns = ["A" + str(i).zfill(4) for i in range(1, num_annotators+1)]
    # Drop rows not annotated by anybody
    row_NA_check = pd.notna(s).any(axis=1)
    X_train = X_train[row_NA_check]
    true_labels_train = true_labels_train[row_NA_check]
    multiannotator_labels = s[row_NA_check].reset_index(drop=True)
    # Shuffle the rows of the dataset
    shuffled_indices = np.random.permutation(len(X_train))
    return {
        "X_train": X_train[shuffled_indices],
        "true_labels_train": true_labels_train[shuffled_indices],
        "multiannotator_labels": multiannotator_labels.iloc[shuffled_indices],
    }

def generate_noisy_labels(true_labels, annotator_quality):
    """ Randomly flips each true label to a different class with probability that depends on annotator_quality. """
    n = len(true_labels)
    m = np.max(true_labels) + 1  # number of classes
    annotated_labels = np.random.randint(low=0, high=3, size=n)
    correctly_labeled_indices = np.random.random(n) < annotator_quality
    annotated_labels[correctly_labeled_indices] = true_labels[correctly_labeled_indices]
    return annotated_labels

