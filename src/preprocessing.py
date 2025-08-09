'''
PART 1: PRE-PROCESSING
- Tailor the code scaffolding below to load and process the data
- Write the functions below
    - Further info and hints are provided in the docstrings
    - These should return values when called by the main.py
'''

import pandas as pd
import ast

def load_data():
    '''
    Load data from CSV files
    
    Returns:
        model_pred_df (pd.DataFrame): DataFrame containing model predictions
        genres_df (pd.DataFrame): DataFrame containing genre information
    '''
    # Your code here
    model_pred_df = pd.read_csv('/Users/Aditi/Documents/UMD/Classes/INST414-Data_Science_Techniques/Problem Sets/problem-set-3/data/prediction_model_03.csv')
    genres_df = pd.read_csv('/Users/Aditi/Documents/UMD/Classes/INST414-Data_Science_Techniques/Problem Sets/problem-set-3/data/genres.csv')

    return model_pred_df, genres_df

def process_data(model_pred_df, genres_df):
    '''
    Process data to get genre lists and count dictionaries
    
    Returns:
        genre_list (list): List of unique genres
        genre_true_counts (dict): Dictionary of true genre counts
        genre_tp_counts (dict): Dictionary of true positive genre counts
        genre_fp_counts (dict): Dictionary of false positive genre counts

    '''
    # Your code here

    #List of unique genres
    genre_list = genres_df['genre'].tolist()

    ##Setup predicted and actual genres lists for dictionary initialization
    #Kept as series of lists
    model_pred_df['predicted'] = model_pred_df["predicted"].apply(lambda x: [x])
    model_pred_df["actual genres"] = model_pred_df["actual genres"].apply(ast.literal_eval)

    #Dictionary of true genre counts:
    genre_true_counts = {
        g: 0 for g in genre_list
    }

    #Dictionary of true positive genre counts:
    genre_tp_counts = {
        g: 0 for g in genre_list
    }
    
    #Dictionary of false positive genre counts
    genre_fp_counts = {
        g: 0 for g in genre_list
    }

    #Initialize the counting for each dictionary: 
    for _, row in model_pred_df.iterrows():
        actual = row['actual genres']
        predicted = row['predicted']

        for g in genre_list:
            if g in actual:
                genre_true_counts[g] += 1
            if g in predicted and g in actual:
                genre_tp_counts[g] += 1
            if g in predicted and g not in actual:
                genre_fp_counts[g] += 1
  
    return genre_list, genre_true_counts, genre_tp_counts, genre_fp_counts
