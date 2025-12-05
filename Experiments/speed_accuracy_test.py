import numpy as np
import pandas as pd
import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Algorithms.random_svd import randomized_svd
from Algorithms.power_iteration_svd import random_svd_power_iter


def speed_accuracy_test(data, proportion, k, r, q = 0, range_method = 'qr', show_results = False):
    '''
    Test to compare the speed and accuracy of the standard, Random, and Power Iteration SVD Methods.
    
    Parameters:
            data (ndarray) : mxn input matrix
            proportion (float): the percentage of data to include in the randomized methods (the amount you will randomly keep)
            k (int) : target rank
            r (int) : number of subspace iterations
            q (int) : number of power iterations (default is 0)
            range_method (str) : method to compute the range ('qr' or 'subspace_iter')
                qr : use the basic numpy randomized range finder
                subspace_iter : use the randomized subspace iteration method (Algorithm 4.4.2)
            show_results (bool): indicating if you want a table of results printed
        Returns:=
            accuracy (tuple) : ||A - A^hat|| for each estimator
            speed (tuple) : the amount of time it took for each method to complete
    '''
    # Standard SVD
    standard_start = time.time()
    U, S, VH = np.linalg.svd(data)
    standard_accuracy = np.linalg.norm(data - U @ np.diag(S) @ VH)
    standard_time = time.time() - standard_start
    
    # Random SVD
    random_start = time.time()
    U, S, VH = randomized_svd(data, k, r, q, range_method, proportion = proportion)
    random_accuracy = np.linalg.norm(data - U @ np.diag(S) @ VH)
    random_time = time.time() - random_start
    
    # Power Iteration SVD
    power_start = time.time()
    U, S, VH = random_svd_power_iter(data, k, q, proportion = proportion)
    power_accuracy = np.linalg.norm(data - U @ np.diag(S) @ VH)
    power_time = time.time() - power_start
    
    if show_results:
        # Your data
        times = [standard_time, random_time, power_time]
        accuracies = [standard_accuracy, random_accuracy, power_accuracy]
        methods = ["Standard SVD", "Random SVD", "Power Iteration SVD"]

        # Create DataFrame
        df = pd.DataFrame({
            "Method": methods,
            "Time": times,
            "Accuracy": accuracies
        })
        
        print(df)
    
    return times, accuracies



    
    
        