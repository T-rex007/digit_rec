import tensorflow as tf
import numpy as np
from IPython.display import display
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from tensorflow.contrib.layers import fully_connected

data = pd.read_csv('train.csv')
features = np.array(data.drop('label', axis = 1))
labels = OneHotEncoder().fit_transform(data.label.values.reshape(-1,1)).todense()
features = StandardScaler().fit_transform(np.float32(features))
xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size = .25,
                                                shuffle = True, stratify = labels,
                                                random_state = 42)

n_inputs = features.shape[1]
n_hidden1 = 200
n_hidden2 = 100
n_outputs = labels.shape[1]

sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, shape = [ None, n_inputs], name = "x")
yt = tf.placeholder(tf.float32, shape = [None, n_outputs], name = "ytrue")

### Defining a neural layer
def neuron_layer(x, n_neurons, name, activation = None):
    with tf.name_scope(name):
        n_inputs = int(x.get_shape()[1])
        stddev = 2/ np.sqrt(n_inputs)
        init = tf.truncated_normal((n_inputs, n_neurons), stddev = stddev)
        w = tf.Variable(init, name = "weights")
        b = tf.Variable(tf.zeros([n_neurons], tf.float32))
        z = tf.add(tf.matmul(x,w),b)
        if activation == "relu":
            return tf.nn.relu(z)
        elif activation == "softmax": 
            return tf.nn.softmax(z)
        else:
            return z
    
with tf.name_scope("dnn"):
    hidden1 = fully_connected(x, n_hidden1, scope = 'h1')
    hidden2 = fully_connected(hidden1, n_hidden2, scope = 'h2')
    logits = fully_connected( hidden2, n_outputs,scope = 'outputs', activation_fn = None)

sess.run(tf.global_variables_initializer())

cross_entropy = tf.reduce_mean(-tf.reduce_sum(yt * tf.log(logits), reduction_indices = [1]))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)

def generator(features, labels, batch_size=50 ):
    for i in range(0, len(features), batch_size):
        yield (features[i:i+batch_size], labels[i:i+batch_size])

epochs = 5
for n in range(epochs):
    train_gen = generator(xtrain, ytrain)
    for i in range(len(xtrain)//50):
        xtr, ytr = next(train_gen)
        train_step.run(feed_dict = {x: xtr, yt: ytr})

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(yt,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
acc = accuracy.eval(feed_dict = {x: xtest, yt: ytest})* 100
print(acc)