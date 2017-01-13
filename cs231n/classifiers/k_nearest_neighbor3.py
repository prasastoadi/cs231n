import numpy as np

class KNearestNeighbor3(object):
    """ K Nearest Neighbor classifier with L2 distance"""

    def __init__(self):
        pass

    def train(self, train_images, train_labels):
        """
        Train the classifier. 

        Inputs:
        - train_image: A numpy array of shape (num_train, D)
        - train_label: A numpy array of shape (num_label,).
          train_label[i] is the label for train_image[i]
        """
        self.train_images = train_images
        self.train_labels = train_labels

    def predict(self, test_images, k=1, num_loops=0):
        """
        Predict labels for test data using this classifier

        Inputs:
        - test_image: A numpy array of shape (num_test, D) containing test
          of num_test samples each of dimension D
        - k: The number of nearest neighbors that vote for the prediciton
        - num_loops: Determines which implementation to use to compute distances
                     between training points and testing points.

        Returns:
        - test_label: A numpy array of shape(num_test,) containing predicted label
                      of the test data  
        """
        if num_loops == 0:
            dists = self.compute_distances_no_loops(test_images)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(test_images)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(test_images)
        else:
            raise ValueError('Invalid value %d for num_loops'% num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, test_images):
        """
        Inputs:
        - test_images: A numpy array of shape(num_test, D) containing test data.

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
        is the Euclidean distance between the ith test point and the jth training point.
        """

        num_test = test_images.shape[0]
        num_train = self.train_images.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train): 
                dists[i, j] = np.linalg.norm(test_images[i]-self.train_images[j])
        return dists

    def compute_distances_one_loop(self, test_images):
        """
        Inputs:
        - test_images: A numpy array of shape(num_test, D) containing test data.

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
        is the Euclidean distance between the ith test point and the jth training point.
        """
        num_test = test_images.shape[0]
        num_train = self.train_images.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            dists[i,:] = np.linalg.norm(test_images[i]-self.train_images, axis=1)
        return dists

    def compute_distances_no_loops(self, test_images):
        """
        Inputs:
        - test_images: A numpy array of shape(num_test, D) containing test data.

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
        is the Euclidean distance between the ith test point and the jth training point.
        """
        num_test = test_images.shape[0]
        num_train = self.train_images.shape[0]
        #dists = np.linalg.norm(test_images[:,np.newaxis,:] - self.train_images[np.newaxis,:,:], axis=2)
        
        sum_of_square_test = np.square(test_images).sum(axis=1)[:,np.newaxis]
        sum_of_square_train = np.square(self.train_images).sum(axis=1)
        dists = np.sqrt(sum_of_square_test - 2*np.dot(test_images,self.train_images.T) + sum_of_square_train)

        return dists

    def predict_labels(self, dists, k=1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.

        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance betwen the ith test point and the jth training point.

        Returns:
        - labels_pred: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].  
        """
        num_test = dists.shape[0]
        labels_pred = np.zeros(num_test)
        for i in range(num_test):
            # A list of length k storing the labels of the k nearest neighbors to
            # the ith test point.
            k_sorted = np.argsort(dists[i])[:k]
            k_labels = self.train_labels[k_sorted]
            unique, pos = np.unique(k_labels, return_inverse=True)
            labels_pred[i] = unique[np.bincount(pos).argmax()]

        return labels_pred