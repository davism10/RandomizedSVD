import numpy as np
import pandas as pd
import time
import sys
import os
from matplotlib import pyplot as plt
from tqdm import tqdm


from speed_accuracy_test import *


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def best_rank(data, proportion, oversamples, r, q, range_method):
    '''
    Test to compare the target rank to reconstruction accuracy for Random SVD.
    
    '''
    ranks = np.arange(1, len(data), 20)
    print(ranks)
    results = np.array([speed_accuracy_test(data, proportion, oversamples, k, r, q, range_method) for k in tqdm(ranks)])
    times = results[:,0]
    ac = results[:,1]
    labels = ["Standard", "Random", f"Power ({q})"]
    colors = ['hotpink', 'mediumpurple', 'royalblue']
    
    fig = plt.figure()
    fig.figsize = (7,7)
    fig.patch.set_facecolor('#ffc7d4')
    # ax = plt.axes()
    # ax.set_facecolor('#ffc7d4')
    plt.subplot(2,1,1)
    for i in range(3):
        plt.plot(ranks, times[:,i], label = labels[i], color = colors[i])
        plt.title("Time to Compute SVD")
        plt.xlabel("Rank")
        plt.ylabel("Time")
        plt.legend()
        
    plt.subplot(2,1,2)
    for i in range(3):
        plt.plot(ranks, ac[:,i], label = labels[i], color = colors[i])
        plt.title("SVD Estimation Accuracy")
        plt.xlabel("Rank")
        plt.ylabel("Accuracy")
        plt.legend()
        
    plt.suptitle("Randomized SVD Error Metrics")
    plt.tight_layout()
    plt.savefig('rank.png')
    
    plt.subplot(2,1,1)
    for i in range(1,3):
        plt.plot(ranks, times[:,i], label = labels[i], color = colors[i])
        plt.title("Time to Compute SVD")
        plt.xlabel("Rank")
        plt.ylabel("Time")
        plt.legend()
        
    plt.subplot(2,1,2)
    for i in range(1,3):
        plt.plot(ranks, ac[:,i], label = labels[i], color = colors[i])
        plt.title("SVD Estimation Accuracy")
        plt.xlabel("Rank")
        plt.ylabel("Accuracy")
        plt.legend()
        
    plt.suptitle("Randomized SVD Error Metrics")
    plt.tight_layout()
    plt.savefig('rank_.png')
        

def best_subspace_iteration(data, proportion, oversamples, k, q = 0, range_method = 'qr'):
    iterations = np.arange(1, 100, 3)
    results = np.array([speed_accuracy_test(data, proportion, oversamples, k, r, q, range_method) for r in tqdm(iterations)])
    times = results[:,0]
    ac = results[:,1]
    labels = ["Standard", "Random", f"Power ({q})"]
    colors = ['hotpink', 'mediumpurple', 'royalblue']
    
    fig = plt.figure()
    fig.figsize = (7,7)
    fig.patch.set_facecolor('#ffc7d4')
    # ax = plt.axes()
    # ax.set_facecolor('#ffc7d4')
    plt.subplot(2,1,1)
    for i in range(3):
        plt.plot(iterations, times[:,i], label = labels[i], color = colors[i])
        plt.title("Time to Compute SVD")
        plt.xlabel("Number of Subspace Iterations")
        plt.ylabel("Time")
        plt.legend()
        
    plt.subplot(2,1,2)
    for i in range(3):
        plt.plot(iterations, ac[:,i], label = labels[i], color = colors[i])
        plt.title("SVD Estimation Accuracy")
        plt.xlabel("Number of Subspace Iterations")
        plt.ylabel("Accuracy")
        plt.legend()
        
    plt.suptitle("Randomized SVD Error Metrics")
    plt.tight_layout()
    plt.savefig('subspace_iters.png')
    
    fig = plt.figure()
    fig.figsize = (7,7)
    fig.patch.set_facecolor('#ffc7d4')
    
    plt.subplot(2,1,1)
    for i in range(1,3):
        plt.plot(iterations, times[:,i], label = labels[i], color = colors[i])
        plt.title("Time to Compute SVD")
        plt.xlabel("Number of Subspace Iterations")
        plt.ylabel("Time")
        plt.legend()
        
    plt.subplot(2,1,2)
    for i in range(1,3):
        plt.plot(iterations, ac[:,i], label = labels[i], color = colors[i])
        plt.title("SVD Estimation Accuracy")
        plt.xlabel("Number of Subspace Iterations")
        plt.ylabel("Accuracy")
        plt.legend()
        
    plt.suptitle("Randomized SVD Error Metrics")
    plt.tight_layout()
    plt.savefig('subspace_iters_.png')


def best_power_iterations(data, proportion, oversamples, k, r, range_method = 'qr'):
    iterations = np.arange(1, 100, 3)
    results = np.array([speed_accuracy_test(data, proportion, oversamples, k, r, q, range_method) for q in tqdm(iterations)])
    times = results[:,0]
    ac = results[:,1]
    labels = ["Standard", "Random", f"Power"]
    colors = ['hotpink', 'mediumpurple', 'royalblue']
    
    fig = plt.figure()
    fig.figsize = (7,7)
    fig.patch.set_facecolor('#ffc7d4')

    plt.subplot(2,1,1)
    for i in range(3):
        plt.plot(iterations, times[:,i], label = labels[i], color = colors[i])
        plt.title("Time to Compute SVD")
        plt.xlabel("Number of Power Iterations")
        plt.ylabel("Time")
        plt.legend()
        
    plt.subplot(2,1,2)
    for i in range(3):
        plt.plot(iterations, ac[:,i], label = labels[i], color = colors[i])
        plt.title("SVD Estimation Accuracy")
        plt.xlabel("Number of Power Iterations")
        plt.ylabel("Accuracy")
        plt.legend()
        
    plt.suptitle("Randomized SVD Error Metrics")
    plt.tight_layout()
    plt.savefig('power_iters.png')
    
    
    fig = plt.figure()
    fig.figsize = (7,7)
    fig.patch.set_facecolor('#ffc7d4')

    plt.subplot(2,1,1)
    for i in range(1,3):
        plt.plot(iterations, times[:,i], label = labels[i], color = colors[i])
        plt.title("Time to Compute SVD")
        plt.xlabel("Number of Power Iterations")
        plt.ylabel("Time")
        plt.legend()
        
    plt.subplot(2,1,2)
    for i in range(1,3):
        plt.plot(iterations, ac[:,i], label = labels[i], color = colors[i])
        plt.title("SVD Estimation Accuracy")
        plt.xlabel("Number of Power Iterations")
        plt.ylabel("Accuracy")
        plt.legend()
        
    plt.suptitle("Randomized SVD Error Metrics")
    plt.tight_layout()
    plt.savefig('power_iters_.png')
    
    
def best_proportion(data, oversamples, k, r, q = 0, range_method = 'qr'):
    proportions = np.linspace(.1,1,50)
    results = np.array([speed_accuracy_test(data, proportion, oversamples, k, r, q, range_method) for proportion in tqdm(proportions)])
    times = results[:,0]
    ac = results[:,1]
    labels = ["Standard", "Random", f"Power ({q})"]
    colors = ['hotpink', 'mediumpurple', 'royalblue']
    
    fig = plt.figure()
    fig.figsize = (7,7)
    fig.patch.set_facecolor('#ffc7d4')
    # ax = plt.axes()
    # ax.set_facecolor('#ffc7d4')
    plt.subplot(2,1,1)
    for i in range(3):
        plt.plot(proportions, times[:,i], label = labels[i], color = colors[i])
        plt.title("Time to Compute SVD")
        plt.xlabel("Proportion of Data Kept")
        plt.ylabel("Time")
        plt.legend()
        
    plt.subplot(2,1,2)
    for i in range(3):
        plt.plot(proportions, ac[:,i], label = labels[i], color = colors[i])
        plt.title("SVD Estimation Accuracy")
        plt.xlabel("Proportion of Data Kept")
        plt.ylabel("Accuracy")
        plt.legend()
        
    plt.suptitle("Randomized SVD Error Metrics")
    plt.tight_layout()
    plt.savefig('proportion_iters.png')
    
    fig = plt.figure()
    fig.figsize = (7,7)
    fig.patch.set_facecolor('#ffc7d4')
    # ax = plt.axes()
    # ax.set_facecolor('#ffc7d4')
    plt.subplot(2,1,1)
    for i in range(1,3):
        plt.plot(proportions, times[:,i], label = labels[i], color = colors[i])
        plt.title("Time to Compute SVD")
        plt.xlabel("Proportion of Data Kept")
        plt.ylabel("Time")
        plt.legend()
        
    plt.subplot(2,1,2)
    for i in range(1,3):
        plt.plot(proportions, ac[:,i], label = labels[i], color = colors[i])
        plt.title("SVD Estimation Accuracy")
        plt.xlabel("Proportion of Data Kept")
        plt.ylabel("Accuracy")
        plt.legend()
        
    plt.suptitle("Randomized SVD Error Metrics")
    plt.tight_layout()
    plt.savefig('proportion_iters_.png')


def best_oversamples(data, proportion, k, r, q = 0, range_method = 'qr'):
    oversamples = np.arange(1,len(data) - k, (len(data)-1-k)//20)
    results = np.array([speed_accuracy_test(data, proportion, oversample, k, r, q, range_method) for oversample in tqdm(oversamples)])
    times = results[:,0]
    ac = results[:,1]
    labels = ["Standard", "Random", f"Power ({q})"]
    colors = ['hotpink', 'mediumpurple', 'royalblue']
    
    fig = plt.figure()
    fig.figsize = (7,7)
    fig.patch.set_facecolor('#ffc7d4')
    # ax = plt.axes()
    # ax.set_facecolor('#ffc7d4')
    plt.subplot(2,1,1)
    for i in range(3):
        plt.plot(oversamples, times[:,i], label = labels[i], color = colors[i])
        plt.title("Time to Compute SVD")
        plt.xlabel(f"Number of samples kept greater after {k}")
        plt.ylabel("Time")
        plt.legend()
        
    plt.subplot(2,1,2)
    for i in range(3):
        plt.plot(oversamples, ac[:,i], label = labels[i], color = colors[i])
        plt.title("SVD Estimation Accuracy")
        plt.xlabel(f"Number of samples kept greater after {k}")
        plt.ylabel("Accuracy")
        plt.legend()
        
    plt.suptitle("Randomized SVD Error Metrics")
    plt.tight_layout()
    plt.savefig('oversamples.png')
    
    fig = plt.figure()
    fig.figsize = (7,7)
    fig.patch.set_facecolor('#ffc7d4')
    # ax = plt.axes()
    # ax.set_facecolor('#ffc7d4')
    plt.subplot(2,1,1)
    for i in range(1,3):
        plt.plot(oversamples, times[:,i], label = labels[i], color = colors[i])
        plt.title("Time to Compute SVD")
        plt.xlabel(f"Number of samples kept greater after {k}")
        plt.ylabel("Time")
        plt.legend()
        
    plt.subplot(2,1,2)
    for i in range(1,3):
        plt.plot(oversamples, ac[:,i], label = labels[i], color = colors[i])
        plt.title("SVD Estimation Accuracy")
        plt.xlabel(f"Number of samples kept greater after {k}")
        plt.ylabel("Accuracy")
        plt.legend()
        
    plt.suptitle("Randomized SVD Error Metrics")
    plt.tight_layout()
    plt.savefig('oversamples_.png')


def comp_range_method(data, proportion, oversamples, k, r, q = 0, range_method = 'qr'):
    range_methods = ['qr', 'subspace_iter']
    results = np.array([speed_accuracy_test(data, proportion, oversamples, k, r, q, range_method) for range_method in tqdm(range_methods)])
    times = results[:,0]
    ac = results[:,1]
    labels = ["Standard", "Random", f"Power ({q})"]
    colors = ['hotpink', 'mediumpurple', 'royalblue']
    
    fig = plt.figure()
    fig.figsize = (7,7)
    fig.patch.set_facecolor('#ffc7d4')
    # ax = plt.axes()
    # ax.set_facecolor('#ffc7d4')
    
    plt.subplot(2,1,1)
    for i in range(3):
        plt.plot(times[:,i], label = labels[i], color = colors[i])
        plt.title("Time to Compute SVD")
        plt.xlabel(f"Range Method")
        plt.ylabel("Time")
        plt.legend()
        
    plt.subplot(2,1,2)
    for i in range(3):
        plt.plot(ac[:,i], label = labels[i], color = colors[i])
        plt.title("SVD Estimation Accuracy")
        plt.xlabel(f"Range Method")
        plt.ylabel("Accuracy")
        plt.legend()
        
    plt.suptitle("Randomized SVD Error Metrics")
    plt.tight_layout()
    plt.savefig('range_method.png')
    
    
    fig = plt.figure()
    fig.figsize = (7,7)
    fig.patch.set_facecolor('#ffc7d4')
    # ax = plt.axes()
    # ax.set_facecolor('#ffc7d4')
    
    plt.subplot(2,1,1)
    for i in range(1,3):
        plt.plot(times[:,i], label = labels[i], color = colors[i])
        plt.title("Time to Compute SVD")
        plt.xlabel(f"Range Method")
        plt.ylabel("Time")
        plt.legend()
        
    plt.subplot(2,1,2)
    for i in range(1,3):
        plt.plot(ac[:,i], label = labels[i], color = colors[i])
        plt.title("SVD Estimation Accuracy")
        plt.xlabel(f"Range Method")
        plt.ylabel("Accuracy")
        plt.legend()
        
    plt.suptitle("Randomized SVD Error Metrics")
    plt.tight_layout()
    plt.savefig('range_method_.png')

def test_size(size):
    data = np.random.rand(size,size)
    
    # gridsearch parameters
    n = 3
    proportions = list(np.linspace(.1,1,n))
    k = np.arange(10, len(data), (len(data) - 10)//n)
    r = np.arange(1, 100, 99//n)
    q = np.arange(1, 100, 99//n)
    range_methods = ['qr', 'subspace_iter']
    
    proportions.append(None)
    
    best_accuracy = np.inf
    best_a_vals = []
    best_time = 0
    
    for prop in tqdm(proportions):
        for k_i in k:
            oversamples = list(np.arange(1,len(data) - k_i, (len(data) - k_i - 20)//n))
            oversamples.append(None)
            for r_i in r:
                for q_i in q:
                    for oversample in oversamples:
                        for range_meth in range_methods:
                            time, accuracy = speed_accuracy_test(data, prop, oversample, k_i, r_i, q_i, range_meth, standard = False, power = True, random = False)
                            if accuracy[0] < best_accuracy:
                                print(f"New Best Accuracy: {accuracy}", flush=True)
                                best_accuracy = accuracy[0]
                                best_time = time[0]
                                best_a_vals = [prop, k_i, r_i, q_i, oversample, range_meth]
    
    print("Best:")
    print(f"accuracy: {best_accuracy}")
    print(f"values: {best_a_vals}")
    print(f"time: {best_time}")
          
                            
def mat_size(r = 2, q = 0, range_method = 'qr', proportion = None, oversamples = None, plot = False):
    sizes =  np.arange(20,2000,50)
    results = np.array([speed_accuracy_test(np.random.rand(size,size), proportion, oversamples, 10, r, q, range_method) for size in tqdm(sizes)])
    times = results[:,0]
    ac = results[:,1]
    labels = ["Standard", "Random", f"Power ({q})"]
    colors = ['hotpink', 'mediumpurple', 'royalblue']
    
    fig = plt.figure()
    fig.figsize = (7,7)
    fig.patch.set_facecolor('#ffc7d4')
    # ax = plt.axes()
    # ax.set_facecolor('#ffc7d4')
    plt.subplot(2,1,1)
    for i in range(3):
        plt.plot(sizes, times[:,i], label = labels[i], color = colors[i])
        plt.title("Time to Compute SVD")
        plt.xlabel(f"Matrix Dimension")
        plt.ylabel("Time")
        plt.legend()
        
    plt.subplot(2,1,2)
    for i in range(3):
        plt.plot(sizes, ac[:,i], label = labels[i], color = colors[i])
        plt.title("SVD Estimation Accuracy")
        plt.xlabel(f"Matrix Dimension")
        plt.ylabel("Accuracy")
        plt.legend()
        
    plt.suptitle("Randomized SVD Error Metrics")
    plt.tight_layout()
    plt.savefig('sizes.png')
         
    
    fig = plt.figure()
    fig.figsize = (7,7)
    fig.patch.set_facecolor('#ffc7d4')
    # ax = plt.axes()
    # ax.set_facecolor('#ffc7d4')
    plt.subplot(2,1,1)
    for i in range(1,3):
        plt.plot(sizes, times[:,i], label = labels[i], color = colors[i])
        plt.title("Time to Compute SVD")
        plt.xlabel(f"Matrix Dimension")
        plt.ylabel("Time")
        plt.legend()
        
    plt.subplot(2,1,2)
    for i in range(1,3):
        plt.plot(sizes, ac[:,i], label = labels[i], color = colors[i])
        plt.title("SVD Estimation Accuracy")
        plt.xlabel(f"Matrix Dimension")
        plt.ylabel("Accuracy")
        plt.legend()
        
    plt.suptitle("Randomized SVD Error Metrics")
    plt.tight_layout()
    plt.savefig('sizes_.png')
    

if __name__ == "__main__":
    # data = np.random.rand(1200,1200)
    
    # best_rank(data, proportion = None, oversamples=None, r = 2, q = 2, range_method='qr')
    # print("Rank Done")
    
    # best_subspace_iteration(data, proportion = None, oversamples = None, k = 800, q = 2, range_method = 'qr')
    # print("Subspace Done")
    
    # best_power_iterations(data, proportion = None, oversamples = None, k = 800, r = 2, range_method = 'qr')
    # print("power done")
    
    # best_proportion(data, oversamples = None, k = 800, r = 2, q = 2, range_method = 'qr')
    # print("proportion done")
    
    # best_oversamples(data, proportion = None, k = 800, r = 2, q = 2, range_method = 'qr')
    # print("oversamples done")
    
    # comp_range_method(data, proportion = None, oversamples= None, k = 800, r = 2, q = 2, range_method = 'qr')
    # print("range methods done")
    
    # mat_size(r = 2, q = 2, range_method = 'qr', proportion = None, oversamples = None)
    # print('sizes done')
    
    # # test_size(400)
    # proportion, k, r, q, oversamples, range_method = None, 270, 67, 67, None, 'subspace_iter'
    # data = np.random.rand(400,400)
    
    # best_rank(data, proportion = proportion, oversamples=oversamples, r = r, q = q, range_method=range_method)
    # print("Rank Done")
    
    # best_subspace_iteration(data, proportion = proportion, oversamples = oversamples, k = 800, q = q, range_method = range_method)
    # print("Subspace Done")
    
    # best_power_iterations(data, proportion = proportion, oversamples = oversamples, k = 800, r = r, range_method = range_method)
    # print("power done")
    
    # best_proportion(data, oversamples = oversamples, k = 800, r = r, q = q, range_method = range_method)
    # print("proportion done")
    
    # best_oversamples(data, proportion = proportion, k = 800, r = r, q = q, range_method = range_method)
    # print("oversamples done")
    
    # comp_range_method(data, proportion = proportion, oversamples=oversamples, k = 800, r = r, q = q, range_method = range_method)
    # print("range methods done")
    
    # mat_size(r = r, q = q, range_method = range_method, proportion = proportion, oversamples = oversamples)
    # print('sizes done')
    
    test_size(2000)
    
    
    


    
    
        