import numpy as np
import pandas as pd
import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Algorithms.random_svd import randomized_svd
from Algorithms.power_iteration_svd import random_svd_power_iter


def speed_accuracy_test(data, proportion, oversamples, k, r, q = 0, range_method = 'qr', show_results = False, standard = True, power=True, random=True):
    '''
    Test to compare the speed and accuracy of the standard, Random, and Power Iteration SVD Methods.
    
    Parameters:
            data (ndarray) : mxn input matrix
            proportion (float) or oversamples (int): the proportion of columns to keep in your projection or the number of oversamples to add
                pick one of these two parameters to set the size of the random projection matrix
                if both are None, defaults to oversamples = k
                if both are provided, proportion takes precedence
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
    times = []
    accuracies = []
    # Standard SVD
    if standard:
        standard_start = time.time()
        U, S, VH = np.linalg.svd(data)
        standard_accuracy = np.linalg.norm(data - U[:,:k] @ np.diag(S[:k]) @ VH[:k,:])
        standard_time = time.time() - standard_start
        times.append(standard_time)
        accuracies.append(standard_accuracy)
    
    # Random SVD
    if random:
        random_start = time.time()
        U, S, VH = randomized_svd(data, k, r, 0, range_method, proportion = proportion, oversamples=oversamples)
        random_accuracy = np.linalg.norm(data - U @ np.diag(S) @ VH)
        random_time = time.time() - random_start
        times.append(random_time)
        accuracies.append(random_accuracy)
    
    # Power Iteration SVD
    if power:
        power_start = time.time()
        U, S, VH = randomized_svd(data, k, r, q, range_method, proportion = proportion, oversamples=oversamples)
        power_accuracy = np.linalg.norm(data - U @ np.diag(S) @ VH)
        power_time = time.time() - power_start
        times.append(power_time)
        accuracies.append(power_accuracy)
        

    if show_results:
        # Your data
        methods = ["Standard SVD", "Random SVD", "Power Iteration SVD"]

        # Create DataFrame
        df = pd.DataFrame({
            "Method": methods,
            "Time": times,
            "Accuracy": accuracies
        })
        
        print(df)
    
    return times, accuracies



    
    
        