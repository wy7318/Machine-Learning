import tensorflow as tf
import tensorflow_probability as tfp



'''Hidden Markov Models
Hidden Markov Model is a statistical Markov model in which the system being modeled is assumed to be a Markov process – call it X – with unobservable states

States : i.e. warm, cold, high, low
Observations : i.e. "On a hot day, Matt has a 80% chance of being happy and a 20% change of being sad"
Transitions : i.e. " A cold day has a 30% chance of being followed by a hot day and a 70% chance of being followed by another cold day"
'''

tfd = tfp.distributions
initial_distribution = tfd.Categorical(probs = [0.8, 0.2]) #The first day in our sequence has an 80% chance of being cold
transition_distribution = tfd.Categorical(probs = [[0.7, 0.3], #A cold day has a 30% chance of being followed by a hot day.
                                                   [0.2, 0.8]]) #A hot day has a 20% chance of being followed by a cold day
observation_distribution = tfd.Normal(loc=[0., 15.], scale=[5., 10.]) #On each dat the temp is normally distributed with mean and standard deviation 0 and 5 on a cold day
                                                                      #and mean and standard deviation 15 and 10 on a hot day

#create model
model = tfd.HiddenMarkovModel(
    initial_distribution = initial_distribution,
    transition_distribution = transition_distribution,
    observation_distribution = observation_distribution,
    num_steps=7 # How many times
)

mean = model.mean() # Calculate probability

with tf.compat.v1.Session() as sess:
    print(mean.numpy())

'''result
[2.9999998 5.9999995 7.4999995 8.25      8.625     8.812501  8.90625  ] Print out 7 days temperature prediction
'''
