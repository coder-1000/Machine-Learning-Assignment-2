

#implement logistic regression 
#logistic regression prediction function is y = sigmoid(W^Tx + b)
#train the logistic regression model using SGD and mini batch size B = 500 on the two-class notNMIST dataset
#how to train the dataset:

import tensorflow as tf
import numpy as np


##########################################################
###############Extracting data############################

with np.load("notMNIST.npz") as data :
    Data, Target = data ["images"], data["labels"]
    posClass = 2
    negClass = 9
    dataIndx = (Target==posClass) + (Target==negClass)
    Data = Data[dataIndx]/255.
    Target = Target[dataIndx].reshape(-1, 1)
    Target[Target==posClass] = 1
    Target[Target==negClass] = 0
    np.random.seed(521)
    randIndx = np.arange(len(Data))
    np.random.shuffle(randIndx)
    Data, Target = Data[randIndx], Target[randIndx]
    trainData, trainTarget = Data[:3500], Target[:3500]
    validData, validTarget = Data[3500:3600], Target[3500:3600]
    testData, testTarget = Data[3600:], Target[3600:]

##########################################################
################Defining variables########################

x = tf.placeholder(tf.float32, [784, 500], name = "input_points") #784 dimensions (28x28 pixels) #leave as None in first dimension so that we can choose what to feed to it when we run it
W = tf.Variable(tf.zeros([784,1]), name = "weights") #create a weight vector
b = tf.Variable(tf.constant(1,tf.float32), tf.float32, name = "bias"); #offset
#b = tf.Variable()#(tf.zeros([2])) #what is b?
lambda_ = tf.placeholder(tf.float32) #constant that is fed in at runtime
learning_rate = tf.placeholder(tf.float32) #constant that is fed in at runtime
y = tf.placeholder(tf.float32, name = "target_labels")#this None should be however many data points we pass it 

#########################################################
##############Calculations###############################

weight_squared_sum = tf.matmul(tf.transpose(W),W) #find the square of the weight vector
loss_W = lambda_ /2 * weight_squared_sum #find the loss
y_hat = tf.matmul(tf.transpose(W), x) + b #based on the sigmoid equation given in lab handout #removed the tf.sigmoid part, because the other logits function takes care of it
cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits = y_hat, labels = y) #sigmoid_cross_entropy_with_logits takes in the actual y and the predicted y 
total_loss = tf.reduce_sum(cross_entropy, 1)/500 + loss_W
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss) #change the learning rate each time

#########################################################
#############Training####################################
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

BATCH_SIZE = 500;
NUM_BATCHES = 7;
NUM_ITERATIONS = 5000;

trainX = np.reshape(trainData, (3500, 784));
batchesX = np.array(np.split(trainX, NUM_BATCHES));
batchesY = np.array(np.split(trainTarget, NUM_BATCHES));


with tf.Session() as sess:
	tf.global_variables_initializer().run()
	for i in range(NUM_ITERATIONS*NUM_BATCHES):	
             sess.run(train_step, feed_dict={x:np.transpose(batchesX[i%NUM_BATCHES]), y: batchesY[i%NUM_BATCHES], lambda_:0.01, learning_rate: 0.5});
