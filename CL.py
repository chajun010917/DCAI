import pandas as pd
import numpy as np

#Confident Learning Algorithm 

# Calculating threshold of each class 
def threshold_calc(pred_prob: np.ndarray, labels: np.ndarray) -> np.ndarray:
    n_classes = pred_prob.shape[1]
    thresholds = np.zeros(n_classes)
    for k in range(n_classes):
        class_mask = labels == k #FINDING X_{\tilde{y} = j} setting the mask of the matrix
        class_probs = pred_prob[class_mask, k] #For x \in Previous X 
        thresholds[k] = np.mean(class_probs) if class_probs.size > 0 else 0 #Calculating the mean for threshold
    return thresholds
    

def confident_joint_calc(pred_probs: np.ndarray, labels: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    n_classes = pred_probs.shape[1]
    C = np.zeros((n_classes, n_classes), dtype=int) #Setting up initial matrix C 
    max_indices = np.argmax(pred_probs >= thresholds, axis=1) # Finding indicies in C that contains predicted prob greater than thresholds
    valid_cases = pred_probs[np.arange(pred_probs.shape[0]), max_indices] >= thresholds[max_indices] #Comparing each values
    for i, j in zip(labels[valid_cases], max_indices[valid_cases]): #Setting up confident matrix by looping
        C[i, j] += 1

    return C

def confident_learning(pred_probs: np.ndarray, labels: np.ndarray):
    thresholds = threshold_calc(pred_probs, labels) # Threshold Tj 
    C = confident_joint_calc(pred_probs, labels, thresholds) #Confident joint C 
    num_label_issues = C.sum() - C.trace() #Calculating off diagonal
    self_confidences = np.array([pred_probs[i, l] for i, l in enumerate(labels)]) #This line is excerpted from MIT course lab
    ranked_indices = np.argsort(self_confidences) #Sorting the label errors by model's "Self-Confidence" level
    issue_idx = ranked_indices[:num_label_issues] #Indexes in the given array that contains label errors 
    return issue_idx #Sorting the array by the model's self-confidence is from the lab of the MIT intro to DCAI course. 
    