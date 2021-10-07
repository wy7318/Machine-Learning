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

Hidden Markov Models
Hidden Markov Model is a statistical Markov model in which the system being modeled is assumed to be a Markov process – call it X – with unobservable states

States : i.e. warm, cold, high, low
Observations : i.e. "On a hot day, Matt has a 80% chance of being happy and a 20% change of being sad"
Transitions : i.e. " A cold day has a 30% chance of being followed by a hot day and a 70% chance of being followed by another cold day"
'''
