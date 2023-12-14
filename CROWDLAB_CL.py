import numpy as np
import pandas as pd
import CROWDLAB_Prep
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from CL import confident_learning

# Methods 
def train_model(labels_to_fit, x_train):
    num_crossval_folds = 10  
    model = KNeighborsClassifier(weights="distance")
    pred_probs = cross_val_predict(
        estimator=model, X=x_train, y=labels_to_fit, cv=num_crossval_folds, method="predict_proba"
    )
    return pred_probs

def print_model_accuracy (method, pred_probs, true_labels):
    class_predictions = np.argmax(pred_probs, axis=1)
    held_out_accuracy = np.mean(class_predictions == true_labels)
    print(f"Accuracy of {method} model: {held_out_accuracy}")

# Majority Vote
def get_majority_vote_label(labels_multiannotator, num_classes, pred_probs=None, annotator_quality=None):
    """
    labels_multiannotator: Numpy
    pred_probs: Numpy Array
    annotator_quality: Previously provided from the algorithm for the iterative process
    """

    def get_labels_mode(label_count, num_classes):
        max_count_idx = np.where(label_count == np.nanmax(label_count))[0].astype(float)
        return np.pad(
            max_count_idx, (0, num_classes - len(max_count_idx)), "constant", constant_values=np.NaN
        )

    majority_vote_label = np.full(len(labels_multiannotator), np.nan)
    label_count = np.apply_along_axis(
        lambda s: np.bincount(s[~np.isnan(s)].astype(int), minlength=num_classes),
        axis=1,
        arr=labels_multiannotator,
    )
    mode_labels_multiannotator = np.apply_along_axis(get_labels_mode, axis=1, arr=label_count, num_classes=num_classes)

    nontied_idx = []
    tied_idx = dict()
    # obtaining consensus using annotator majority vote
    for idx, label_mode in enumerate(mode_labels_multiannotator):
        label_mode = label_mode[~np.isnan(label_mode)].astype(int)
        if len(label_mode) == 1:
            majority_vote_label[idx] = label_mode[0]
            nontied_idx.append(idx)
        else:
            tied_idx[idx] = label_mode
            
    # tiebreak 1: using pred_probs:
    if pred_probs is not None and len(tied_idx) > 0:
        for idx, label_mode in tied_idx.copy().items():
            max_pred_probs = np.where(
                pred_probs[idx, label_mode] == np.max(pred_probs[idx, label_mode])
            )[0]
            if len(max_pred_probs) == 1:
                majority_vote_label[idx] = label_mode[max_pred_probs[0]]
                del tied_idx[idx]

    # tiebreak 2: using empirical class frequencies
    if len(tied_idx) > 0:
        class_frequencies = label_count.sum(axis=0)
        for idx, label_mode in tied_idx.copy().items():
            min_frequency = np.where(
                class_frequencies[label_mode] == np.min(class_frequencies[label_mode])
            )[0]
            if len(min_frequency) == 1:
                majority_vote_label[idx] = label_mode[min_frequency[0]]
                del tied_idx[idx]

    # tiebreak 3: using annotator quality scores
    if annotator_quality is None:
        # Calculate initial annotator quality if not provided
        annotator_quality = np.zeros(labels_multiannotator.shape[1])
        for i in range(len(annotator_quality)):
            labels = labels_multiannotator[:, i]
            labels_mask = ~np.isnan(labels)
            if np.sum(labels_mask) == 0:
                annotator_quality[i] = np.NaN
            else:
                annotator_quality[i] = np.mean(
                    labels[labels_mask] == majority_vote_label[labels_mask]
                )

    if len(tied_idx) > 0:
        for idx, label_mode in tied_idx.copy().items():
            label_quality_score = np.array(
                [
                    np.mean(annotator_quality[np.where(labels_multiannotator[idx] == label)[0]])
                    for label in label_mode
                ]
            )
            max_score = np.where(label_quality_score == np.max(label_quality_score))[0]
            if len(max_score) == 1:
                majority_vote_label[idx] = label_mode[max_score[0]]
                del tied_idx[idx]

    # if still tied, break by random selection
    if len(tied_idx) > 0:
        for idx, label_mode in tied_idx.items():
            majority_vote_label[idx] = np.random.choice(label_mode)

    return majority_vote_label.astype(int)

def get_a_MLC(consensus_label: np.ndarray):
    mlc = np.argmax(np.bincount(consensus_label))
    a_MLC = np.mean(consensus_label == mlc)
    return a_MLC

def get_a_m(pred_prob: np.ndarray, consensus_label: np.ndarray):
    a_m = np.mean(np.argmax(pred_prob, axis=1) == consensus_label)
    return a_m

def get_s_j(multiannotator_labels: np.ndarray, num_I: np.ndarray): 
    s_j = np.zeros(multiannotator_labels.shape[1])
    for j in range(len(s_j)):
        annotator_labels_classes = ~np.isnan(multiannotator_labels[:, j])
        ml = multiannotator_labels[annotator_labels_classes]
        s_j_perclass = np.zeros(len(ml))
        total_j = 0
        for i, data in enumerate(ml):
            non_zero_data = data[~np.isnan(data)]
            num_non_zero = len(non_zero_data)
            if num_non_zero > 1: # Meaning that there's more than current j's annotation
                s_j_perclass[i] = (np.sum(non_zero_data == data[j]) - 1) 
                total_j = total_j + num_non_zero - 1
        if np.sum(num_I[annotator_labels_classes] - 1) == 0:
            s_j[j] = np.NaN
        else: 
            s_j[j] = np.sum(s_j_perclass) / total_j
    if np.sum(np.isnan(s_j)) > 0: # If there is an annotator with no agreements, then we will assign 0
        s_j[np.isnan(s_j)] = np.NaN
    return s_j

def get_p_j(multiannotator_labels: np.ndarray, consensus_label: np.ndarray,num_I:np.ndarray ,num_classes):
    annotator_agreement = np.zeros(len(multiannotator_labels))
    for i, labels in enumerate(multiannotator_labels):
        annotator_agreement[i] = np.mean(labels[~np.isnan(labels)] == consensus_label[i])

    num_P = np.mean(annotator_agreement[num_I != 1]) ## P
    neq_ik = (1 - num_P) / (num_classes - 1) # (1-P) / (K-1)
    return num_P, neq_ik

def get_post_pred_probs(
        pred_prob: np.ndarray, 
        multiannotator_labels: np.ndarray, 
        consensus_likelihood, 
        non_consensus_likelihood,
        w_M, 
        w_j,
        num_classes
):
    post_pred_probs = np.zeros(pred_prob.shape)
    for i, labels in enumerate(multiannotator_labels):
        labels_mask = ~np.isnan(labels)
        labels_subset = labels[labels_mask]
        post_pred_probs[i] = [
            np.average(
                [pred_prob[i, true_label]] + [consensus_likelihood if annotator_label == true_label else non_consensus_likelihood
                    for annotator_label in labels_subset], weights=np.concatenate(([w_M], w_j[labels_mask])),
            )for true_label in range(num_classes)
        ]
    return post_pred_probs

def get_annotator_quality (multiannotator_labels, pred_probs, num_I, consensus_label, w_M, s_j):
    multiannotator_labels_subset = multiannotator_labels[num_I != 1]
    consensus_label_subset = consensus_label[num_I != 1]
    
    annotator_label_quality = np.zeros(multiannotator_labels.shape[1])
    annotator_agreement = np.zeros(multiannotator_labels_subset.shape[1])
    for j in range (len(annotator_label_quality)): #Q_j
        labels = multiannotator_labels[:, j]
        labels_mask = ~np.isnan(labels)
        t = labels[labels_mask].astype(int)
        t_pred = pred_probs[labels_mask]
        annotator_label_quality[j] = np.mean(t_pred[np.arange(t.shape[0]), t]) # Label-quality score "self-confidence"
        
    for j in range(len(annotator_agreement)): #A_j
        labels = multiannotator_labels_subset[:, j]
        labels_mask = ~np.isnan(labels)
        if np.sum(labels_mask) == 0:
            annotator_agreement[j] = np.NaN
        else:
            annotator_agreement[j] = np.mean(
                labels[labels_mask] == consensus_label_subset[labels_mask],
            )

    w_0 = np.sum(s_j) * np.mean(num_I) / len(s_j)
    w = w_M / (w_M + w_0)
    annotator_quality = w * annotator_label_quality + (1 - w) * annotator_agreement
    return annotator_quality 

def run_CROWD (multiannotator_labels, num_classes, x_train, pred_prob = None, annotator_quality = None, c_label = None):
    if isinstance(multiannotator_labels, pd.DataFrame):
        multiannotator_labels = multiannotator_labels.replace({pd.NA: np.NaN}).astype(float).to_numpy()

    if c_label is None:
        consensus_label = get_majority_vote_label(multiannotator_labels, num_classes,
                                              pred_probs= pred_prob, annotator_quality=annotator_quality)
    else: consensus_label = c_label  
    if pred_prob is None: pred_prob = train_model(consensus_label, x_train)

    num_I = np.sum(~np.isnan(multiannotator_labels), axis=1)
    num_I_plus = num_I != 1
    cl_I_plus = consensus_label[num_I_plus]
    pred_I_plus = pred_prob[num_I_plus]
    #p_j
    consensus_likelihood, non_consensus_likelihood = get_p_j(multiannotator_labels, consensus_label, num_I, num_classes)

    # Accuracy 
    a_MLC = get_a_MLC(cl_I_plus)
    s_j = get_s_j(multiannotator_labels, num_I)
    a_M = get_a_m(pred_I_plus, cl_I_plus)

    # Weights
    w_j = 1 - ((1-s_j) / (1-a_MLC))
    w_M = (1-((1-a_M)/(1-a_MLC))) * np.sqrt(np.mean(num_I))

    # Post_Pred_Probs
    post_pred_probs = get_post_pred_probs(pred_prob, multiannotator_labels, consensus_likelihood,
                                        non_consensus_likelihood, w_M, w_j, num_classes)

    # Creating new consensus label
    new_consensus_label = np.full(len(consensus_label), np.nan)
    for i in range (len(new_consensus_label)):
        max_prob_ind = np.where(post_pred_probs[i] == np.max(post_pred_probs[i]))[0]
        if len(max_prob_ind) == 1: new_consensus_label[i] = max_prob_ind[0]
        else: new_consensus_label[i] = consensus_label[i]
    new_consensus_label = new_consensus_label.astype(int)

    annotator_quality = get_annotator_quality(multiannotator_labels, pred_prob, num_I, consensus_label, w_M, s_j)
    return new_consensus_label, annotator_quality, post_pred_probs

def run_CROWD_CL (multiannotator_labels, num_classes, x_train, true_labels, itr = 1, pred_prob = None, annotator_quality = None):
    multiannotator_labels = multiannotator_labels.replace({pd.NA: np.NaN}).astype(float).to_numpy()    
    consensus_label = get_majority_vote_label(multiannotator_labels, num_classes, pred_probs= pred_prob, annotator_quality=annotator_quality)
    if pred_prob is None: 
        pred_prob = train_model(consensus_label, x_train)
    for i in range(itr):
        drop_idx = confident_learning(pred_prob, consensus_label)
        keep_idx = ~np.isin(np.arange(x_train.shape[0]), drop_idx)

        x_train = x_train[keep_idx]
        consensus_label = consensus_label[keep_idx]
        multiannotator_labels = multiannotator_labels[keep_idx]
        true_labels = true_labels[keep_idx]

        pred_prob = train_model(consensus_label, x_train)
    print(f"CL + Majority Vote Label {itr} iteration(s): {np.mean(consensus_label == true_labels)}")
    print_model_accuracy(f"CL + Majority Vote {itr} iteration(s)", pred_prob, true_labels)
    a,b,c = run_CROWD(multiannotator_labels, num_classes, x_train, c_label=consensus_label)
    return a,b,c,true_labels, x_train, multiannotator_labels

    
#Multi-annotation Data Creation
data_dict = CROWDLAB_Prep.make_data(sample_size = 5000)
x_train = data_dict["X_train"]
multiannotator_labels = data_dict["multiannotator_labels"]
true_labels = data_dict["true_labels_train"] # used for comparing the accuracy of consensus labels
np_label_mv = multiannotator_labels.replace({pd.NA: np.NaN}).astype(float).to_numpy()

# Creating random consensus label 
labels_from_random_annotators = true_labels.copy()
for i in range(len(multiannotator_labels)):
    annotations_for_example_i = multiannotator_labels.iloc[i][pd.notna(multiannotator_labels.iloc[i])]
    labels_from_random_annotators[i] = np.random.choice(annotations_for_example_i.values)

# Random
print(f"Accuracy of random annotators' Label: {np.mean(labels_from_random_annotators == true_labels)}")
pred_probs_from_model_fit_to_random_annotators = train_model(labels_from_random_annotators, x_train)
num_classes = pred_probs_from_model_fit_to_random_annotators.shape[1]
print_model_accuracy("Random Annotator", pred_probs_from_model_fit_to_random_annotators, true_labels)

# Majority Vote 
label_MV = get_majority_vote_label(np_label_mv, num_classes)
print(f"Accuracy of Majority Vote Label: {np.mean(label_MV == true_labels)}")
print_model_accuracy("Majority-Vote", train_model(label_MV, x_train), true_labels)

# CROWDLAB 
label_CR, aq_CR, prob_CR = run_CROWD(multiannotator_labels, num_classes, x_train)
print(f"Accuracy of CROWDLAB Label: {np.mean(label_CR == true_labels)}")
print_model_accuracy("CROWDLAB", train_model(label_CR, x_train), true_labels)

# CROWDLAB iteration-2
label_CR2, aq_CR2, prob_CR2 = run_CROWD(multiannotator_labels, num_classes, x_train, c_label=label_CR)
print(f"Accuracy of CROWDLAB it-2 Label: {np.mean(label_CR2 == true_labels)}")
print_model_accuracy("CROWDLAB it-2", train_model(label_CR2, x_train), true_labels)

# CL + CROWDLAB
label_CLCR, aq_CLCR, prob_CLCR, tl_CLCR, x_CLCR, _ = run_CROWD_CL(multiannotator_labels, num_classes, x_train, true_labels)
print(f"Accuracy of CL + CROWDLAB Label: {np.mean(label_CLCR == tl_CLCR)}")
print_model_accuracy("CL + CROWDLAB", train_model(label_CLCR, x_CLCR), tl_CLCR)

# Multiple iterations of CL + CROWDLAB
label_mCLCR, aq_mCLCR, prob_mCLCR, tl_mCLCR, x_mCLCR, mutiannotator_mCLCR = run_CROWD_CL(multiannotator_labels, num_classes, x_train, true_labels, itr = 3)
print(f"Accuracy of CL3 + CROWDLAB iterations Label: {np.mean(label_mCLCR == tl_mCLCR)}")
print_model_accuracy("CL3 + CROWDLAB iteration", train_model(label_mCLCR, x_mCLCR), tl_mCLCR)

# Multiple iterations of CL + CROWDLAB it-2
label_mCLCR2, aq_mCLCR2, prob_mCLCR2 = run_CROWD(mutiannotator_mCLCR, num_classes, x_mCLCR, pred_prob= prob_mCLCR, c_label=label_mCLCR)
print(f"Accuracy of CL3 + CROWDLAB2 iterations Label: {np.mean(label_mCLCR2 == tl_mCLCR)}")
print_model_accuracy("CL3 + CROWDLAB2 iteration", train_model(label_mCLCR2, x_mCLCR), tl_mCLCR)

# CROWDLAB Opensource
from cleanlab.multiannotator import get_label_quality_multiannotator
from cleanlab.multiannotator import get_majority_vote_label

label_MV_CRO = get_majority_vote_label(multiannotator_labels)
prob_MV_CRO = train_model(label_MV_CRO, x_train)
results = get_label_quality_multiannotator(multiannotator_labels, prob_MV_CRO, verbose=False)
crowdlab_labels = results["label_quality"]["consensus_label"]
prob_CRO = train_model(crowdlab_labels, x_train)

print(f"Accuracy of CROWDLAB open source-1 label: {np.mean(crowdlab_labels == true_labels)}")
print_model_accuracy("CROWDLAB open source-1", prob_CRO, true_labels)

results2 = get_label_quality_multiannotator(multiannotator_labels, prob_CRO, verbose=False)
crowdlab_labels2 = results2["label_quality"]["consensus_label"]
prob_CRO2 = train_model(crowdlab_labels2, x_train)
print(f"Accuracy of CROWDLAB open source-2 label: {np.mean(crowdlab_labels2 == true_labels)}")
print_model_accuracy("CROWDLAB open source-2", prob_CRO2, true_labels)