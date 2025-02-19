from sklearn.metrics import roc_auc_score
import json
import matplotlib.pyplot as plt
import numpy as np

def calculate_ece(y_true, y_scores, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_scores, bins) - 1
    ece = 0.0
    for i in range(n_bins):
        bin_mask = bin_indices == i
        if np.any(bin_mask):
            bin_accuracy = np.mean([y_true[j] for j in range(len(y_true)) if bin_mask[j]])
            bin_confidence = np.mean([y_scores[j] for j in range(len(y_scores)) if bin_mask[j]])
            bin_size = np.sum(bin_mask) / len(y_true)
            ece += np.abs(bin_accuracy - bin_confidence) * bin_size
    return ece


def evaluate(data):
    try:
        is_correct_list = [result['is_correct'] for result in data]
    except Exception as e:
        is_correct_list = None

    try:
        is_correct_c_list = [result['is_correct_c'] for result in data]
    except Exception as e:
        is_correct_c_list = None

    try:
        scores = [result['consistency_score'] for result in data]
    except Exception as e:
        scores = None

    try:
        scores_c = [result['consistency_score_c'] for result in data]
    except Exception as e:
        scores_c = None

    try:
        auc = round(roc_auc_score(is_correct_list, scores) * 100, 2) if is_correct_list and scores else None
    except Exception as e:
        auc = None

    try:
        auc_c = round(roc_auc_score(is_correct_c_list, scores_c) * 100, 2) if is_correct_c_list and scores_c else None
    except Exception as e:
        auc_c = None

    try:
        accuracy = round((sum(is_correct_list) / len(is_correct_list)) * 100, 2) if is_correct_list else None
    except Exception as e:
        accuracy = None

    try:
        accuracy_c = round((sum(is_correct_c_list) / len(is_correct_c_list)) * 100, 2) if is_correct_c_list else None
    except Exception as e:
        accuracy_c = None

    try:
        ece = round(calculate_ece(is_correct_list, scores) * 100, 2) if is_correct_list and scores else None
    except Exception as e:
        ece = None

    try:
        ece_c = round(calculate_ece(is_correct_c_list, scores_c) * 100, 2) if is_correct_c_list and scores_c else None
    except Exception as e:
        ece_c = None

    return auc, auc_c, accuracy, accuracy_c, ece, ece_c


def evaluate_inference(data):
    is_correct_list = [result['is_correct'] for result in data]
    scores = [result['confidence'] for result in data]
    auc = round(roc_auc_score(is_correct_list, scores) * 100, 2)
    accuracy = round((sum(is_correct_list) / len(is_correct_list)) * 100, 2)
    ece = round(calculate_ece(is_correct_list, scores) * 100, 2)
    return auc, accuracy, ece