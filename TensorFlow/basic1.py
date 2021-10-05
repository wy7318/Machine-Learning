import tensorflow as tf

print(tf.version)

t = tf.zeros([5,5,5,5]) #Number of Element = 5^4
                        
print(t)
t1 = tf.reshape(t, [25, 5, 5]) #Reshaping to rank 25
print(t1)
