import tensorflow as tf
import tensorflow_probability as tfp



'''Clustering
Unsupervised learning algorithm
when only have inputs

K-Means Algorithm:
1) Randomly pick K Points to place K Centroids 
2) Assign each K point to the closest K centroid.
3) Move the K centroid to the center of the mask (group of K Points). 
4) Repeat the action to make K centroid being in the middle 
5) When new K point exist, compare the distances between all K centroid and assign it to the most closest one

'''
