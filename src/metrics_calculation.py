'''
PART 2: METRICS CALCULATION
- Tailor the code scaffolding below to calculate various metrics
- Write the functions below
    - Further info and hints are provided in the docstrings
    - These should return values when called by the main.py

NOTE: Added if __ > 0 else 0 because I got division by 0 errors when I first ran main.py
'''

import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd

def calculate_metrics(model_pred_df, genre_list, genre_true_counts, genre_tp_counts, genre_fp_counts):
    '''
    Calculate micro and macro metrics
    
    Args:
        model_pred_df (pd.DataFrame): DataFrame containing model predictions
        genre_list (list): List of unique genres
        genre_true_counts (dict): Dictionary of true genre counts
        genre_tp_counts (dict): Dictionary of true positive genre counts
        genre_fp_counts (dict): Dictionary of false positive genre counts
    
    Returns:
        tuple: Micro precision, recall, F1 score
        lists of macro precision, recall, and F1 scores
    
    Hint #1: 
    tp -> true positives
    fp -> false positives
    tn -> true negatives
    fn -> false negatives

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    Hint #2: Micro metrics are tuples, macro metrics are lists

    '''
    # Your code here
    sum_tp = sum(genre_tp_counts.values())
    sum_fp = sum(genre_fp_counts.values())
    #Actual calculation for false negative (genre present - model correctly guessed genre = model missed genre)
    sum_fn = sum(genre_true_counts[g] - genre_tp_counts[g] for g in genre_list)

    #Calculate micro precision, recall and F1-score
    micro_precision = sum_tp / (sum_tp + sum_fp) if (sum_tp + sum_fp) > 0 else 0
    micro_recall = sum_tp / (sum_tp + sum_fn) if (sum_tp + sum_fn) > 0 else 0
    micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0

    # Calculate macro metrics per genre, append as lists
    macro_prec_list = []
    macro_recall_list = []
    macro_f1_list = []

    for g in genre_list:
        tp = genre_tp_counts[g]
        fp = genre_fp_counts[g]
        fn = genre_true_counts[g] - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fp) > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        macro_prec_list.append(precision)
        macro_recall_list.append(recall)
        macro_f1_list.append(f1)

    return micro_precision, micro_recall, micro_f1, macro_prec_list, macro_recall_list, macro_f1_list

def calculate_sklearn_metrics(model_pred_df, genre_list):
    '''
    Calculate metrics using sklearn's precision_recall_fscore_support.
    
    Args:
        model_pred_df (pd.DataFrame): DataFrame containing model predictions.
        genre_list (list): List of unique genres.
    
    Returns:
        tuple: Macro precision, recall, F1 score, and micro precision, recall, F1 score.
    
    Hint #1: You'll need these two lists
    pred_rows = []
    true_rows = []
    
    Hint #2: And a little later you'll need these two matrixes for sk-learn
    pred_matrix = pd.DataFrame(pred_rows)
    true_matrix = pd.DataFrame(true_rows)
    '''

    # Your code here
    pred_rows = []
    true_rows = []

    #append 
    for _, row in model_pred_df.iterrows():
        actual = row['actual genres']
        predicted = row['predicted']
        
        #Makes it binary for sk-learn
        true_row = [1 if g in actual else 0 for g in genre_list]
        pred_row = [1 if g in predicted else 0 for g in genre_list]

        true_rows.append(true_row)
        pred_rows.append(pred_row)

    true_matrix = pd.DataFrame(true_rows)
    pred_matrix = pd.DataFrame(pred_rows)

    #Calculate the metrics with sklearn and this output is called in main.py
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(true_matrix, pred_matrix, average='macro', zero_division=0)
    micro_prec, micro_rec, micro_f1, _ = precision_recall_fscore_support(true_matrix, pred_matrix, average='micro', zero_division=0)

    return macro_prec, macro_rec, macro_f1, micro_prec, micro_rec, micro_f1
