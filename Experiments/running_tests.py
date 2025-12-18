from limitations_test import *

if __name__ == "__main__":
    data = np.random.rand(2200,2200)
    
    best_rank(data, proportion = None, oversamples=None, r = 2, q = 2, range_method='qr')
    print("Rank Done")
    
    best_subspace_iteration(data, proportion = None, oversamples = None, k = 800, q = 2, range_method = 'qr')
    print("Subspace Done")
    
    best_power_iterations(data, proportion = None, oversamples = None, k = 800, r = 2, range_method = 'qr')
    print("power done")
    
    best_proportion(data, oversamples = None, k = 800, r = 2, q = 2, range_method = 'qr')
    print("proportion done")
    
    best_oversamples(data, proportion = None, k = 800, r = 2, q = 2, range_method = 'qr')
    print("oversamples done")
    
    comp_range_method(data, proportion = None, oversamples= None, k = 800, r = 2, q = 2, range_method = 'qr')
    print("range methods done")
    
    mat_size(r = 2, q = 2, range_method = 'qr', proportion = None, oversamples = None)
    print('sizes done')