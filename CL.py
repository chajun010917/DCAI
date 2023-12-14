import pandas as pd
import numpy as np

#Confident Learning Algorithm 

# Calculating threshold of each class 
def threshold_calc(pred_prob: np.ndarray, labels: np.ndarray) -> np.ndarray:
    n_classes = pred_prob.shape[1]
    thresholds = np.zeros(n_classes)
    for k in range(n_classes):
        class_mask = labels == k
        class_probs = pred_prob[class_mask, k]
        thresholds[k] = np.mean(class_probs) if class_probs.size > 0 else 0
    return thresholds
    

def confident_joint_calc(pred_probs: np.ndarray, labels: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    n_classes = pred_probs.shape[1]
    C = np.zeros((n_classes, n_classes), dtype=int)
    max_indices = np.argmax(pred_probs >= thresholds, axis=1)
    valid_cases = pred_probs[np.arange(pred_probs.shape[0]), max_indices] >= thresholds[max_indices]
    for i, j in zip(labels[valid_cases], max_indices[valid_cases]):
        C[i, j] += 1

    return C

def confident_learning(pred_probs: np.ndarray, labels: np.ndarray):
    thresholds = threshold_calc(pred_probs, labels)
    C = confident_joint_calc(pred_probs, labels, thresholds)
    num_label_issues = C.sum() - C.trace()
    self_confidences = np.array([pred_probs[i, l] for i, l in enumerate(labels)])
    ranked_indices = np.argsort(self_confidences)
    issue_idx = ranked_indices[:num_label_issues]
    return issue_idx
    