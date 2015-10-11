'''
@brief Assignment 1 for Deep Learning @ MVA/Paris-Saclay
@author Gabriel HUANG
@date October 2015

Open in *spyder* for optimal results
Each #%% block can be launched with the Ctrl-Enter shortcut
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# Select interactive mode
plt.ion()

def show_images(**kwargs):
    '''
    Plot images with titles side by side
    
    Example
        show_image(title1=im1, title2=im2)
    '''
    for i, (title, im) in enumerate(kwargs.items()):
        plt.subplot(1, len(kwargs), i+1)
        plt.title(title)        
        plt.imshow(im)
    plt.draw()

#%% Load digits
digit7 = loadmat('digits/digit7.mat')['D'].astype(float)/255.
train = digit7[::2]
test = digit7[1::2]
print 'Train set {}'.format(train.shape)
print 'Test set {}'.format(test.shape)

#%% Mean digit
meanDigit = np.zeros_like(train[0])
for x in train:
    meanDigit += x / float(train.shape[0])

# Mean digit - simpler & faster
meanDigitNumpy = np.mean(train, axis=0)

# Compare Mean
plt.figure('Mean Image')
show_images(numpy=meanDigitNumpy.reshape(28, 28), yours=meanDigit.reshape(28, 28))

#%% Covariance
covDigitsNumpy = np.cov(train.T)

# Estimate covariance
# hint: look into np.outer()
# <HW>
covDigits = np.zeros((train.shape[1], train.shape[1]))
for x in train:
    delta = x - meanDigit
    covDigits += np.outer(delta, delta) / float(train.shape[0])
# </HW>
    
# Compare
plt.figure('Covariance')
show_images(numpy=covDigitsNumpy, yours=covDigits)

#%% PCA
# Each eigvec[:, i] is i-th component
eigval, eigvec = np.linalg.eigh(covDigits)
tmp_eigval, tmp_eigvec_T = zip(*sorted(zip(eigval, eigvec.T), key=lambda (a,b): a, reverse=True))
eigval, eigvec = np.asarray(tmp_eigval), np.asarray(tmp_eigvec_T).T

plt.figure('PCA')
args = {'vec_{}'.format(i): v.reshape((28, 28)) for i, v in enumerate(eigvec.T[:5])}
show_images(**args)

#%% Estimate PCA approximation error
# hint: look into np.dot()
example = 9
errors = []
plt.figure('PCA Reconstruction')
plt.subplot(3, 4, 1)
plt.imshow(test[example].reshape((28, 28)))
plt.title('Original')
for d in xrange(1, 11):
    # <HW>
    coeffs = np.dot(test - meanDigit, eigvec[:, :d])
    reconstructed = meanDigit + np.dot(coeffs, eigvec[:, :d].T)
    error = np.mean(np.sqrt(np.sum((reconstructed-test)**2, axis=1)))
    # </HW>
        
    plt.subplot(3, 4, 1+d)
    plt.imshow(reconstructed[example].reshape((28, 28)))
    plt.title('d={}'.format(1))
    errors.append(error)
    print 'PCA {} components, error {}'.format(d, error)
plt.figure('PCA approximation error')
plt.plot(np.arange(1, 11), errors)
plt.xlabel('d')
plt.ylabel('Reconstruction error')

#%% Implement K-Means
'''
Implement a K-Means algorithm
Must follow syntax kmeans(data, k, iterations) -> centroids, energy
where trainset has shape (n, d)
centroids have shape (k, d)
and energy = 1/n sum(nn) ||x_nn = c_{assign_nn}||**2

hint: look into np.random.choice
'''

# <HW>
def find_closest(data, centroids):
    '''
    Return
        (assign, distance)
        assign: closest centroid to each sample
        distance: corresponding distance
    
    '''
    assign = np.zeros(data.shape[0], dtype=int)
    distance = np.zeros(data.shape[0])
    for nn, sample in enumerate(data):
        delta = centroids - sample
        sample_to_centroids = np.sum(delta**2, axis=1)
        assign[nn] = np.argmin(sample_to_centroids)
        distance[nn] = sample_to_centroids[assign[nn]]
    return assign, distance
    
def kmeans_energy(data, centroids):
    '''
    Given data and centroids, return K-Means energy
    '''
    assign, distance = find_closest(data, centroids)
    return np.sum(distance)
    return energy
    
def kmeans(data, k, iterations, verbose=False):
    '''
    Cluster data into k clusters using K-Means. 
    '''
    # Initialize centroids randomly
    centroids_idx = np.random.choice(np.arange(data.shape[0]), size=k, replace=False)
    centroids = np.asarray([data[idx] for idx in centroids_idx])
    for i in xrange(iterations):
        # Expectation: Find closest centroid
        assign, current_energy = find_closest(data, centroids)
        # Print status
        if verbose:
            print 'Iteration {} Energy {}'.format(i, current_energy)
        # Maximization: Recompute centroids
        for kk, __ in enumerate(centroids):
            centroids[kk] = np.dot(assign==kk, data) / np.sum(assign==kk)
    return centroids, kmeans_energy(data, centroids)
# </HW>    

#%% [HW] K-Means for K=2
centroids, energy = kmeans(train, 2, 10, verbose=True)

#%% Repeat 10 times K-Means
# <HW>
def kmeans_repeat(data, k, iterations, repeat, verbose=False):
    best_centroids, best_energy = None, None
    for r in xrange(repeat):
        centroids, energy = kmeans(data, k, iterations)
        if best_energy is None or energy < best_energy:
            best_centroids, best_energy = centroids, energy
        if verbose:
            print 'Repeat {} Energy {}'.format(r, energy)
    return best_centroids, best_energy
# </HW>

centroids, energy = kmeans_repeat(train, 2, 10, 10, verbose=True)
plt.figure('2 K-Means Clusters')
show_images(c1=centroids[0].reshape((28, 28)), c2=centroids[1].reshape((28, 28)))


#%%
ks = [3, 4, 5, 10, 50, 100]
ks = [3, 10]  # testing

errors = []
plt.figure('K-Means Reconstruction')
for k in xrange(1, 11):
    # <HW>
    centroids, train_error = kmeans(train, k, 10)
    test_assign, test_error = find_closest(test, centroids)
    reconstructed_example = centroids[test_assign[9]]
    # </HW>
        
    plt.subplot(3, 4, 1+d)
    plt.imshow(reconstructed_example.reshape((28, 28)))
    plt.title('d={}'.format(1))
    errors.append(test_error)
    print 'PCA {} components, error {}'.format(d, error)
plt.figure('K-Means approximation error')
plt.plot(np.arange(1, 11), errors)
plt.xlabel('k number of clusters')
plt.ylabel('Energy')

for k in xrange(1, 11):
    centroids = kmeans(train, 50, 10)
    

#%% For console python only
plt.show()
