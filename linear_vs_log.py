import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time


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

PIXELS = 784;#image is 28 by 28
NUM_ITERATIONS = 20000; #20000
LEARNING_RATE = 0.001;
BATCH_SIZE = 500;
NUM_POINTS = 3500;

trainX = np.reshape(trainData, (3500, -1) );
trainY = trainTarget.astype(np.float64);


        
x = tf.placeholder(tf.float64, name="input_points");
y = tf.placeholder(tf.float64, name="targets");

#mse stuff------------------------

#initialize values to all 0
w = tf.Variable( np.zeros((PIXELS,1), np.float64) ); #784 x 1 containing weights
b = tf.Variable(tf.constant(0,tf.float64), tf.float64); #offset


wcross = tf.Variable( np.zeros((PIXELS,1), np.float64) ); #784 x 1 containing weights
bcross = tf.Variable(tf.constant(0,tf.float64), tf.float64); #offset

yhat = tf.add(tf.matmul(x,w), b); 
yhatCross = tf.add(tf.matmul(x,wcross), bcross);

#loss_d = wx + b - y    
loss = tf.reduce_mean((y - yhat)**2)/2  ; #check dimensions
mseClassError = tf.reduce_mean( tf.abs( tf.round(yhat) - y) );

mseClassValues = []
mseValues = []    


#cross entropy stuff----------------

cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits = yhatCross, labels = y)

crossLoss = tf.reduce_mean(cross_entropy);
crossClassError = tf.reduce_mean(tf.cast(tf.equal(y, tf.round(tf.sigmoid(yhatCross))), tf.float64))
#tf.reduce_mean( tf.abs( tf.round(yhatCross) - y) );

crossClassValues = []
crossValues = []

#set up optimizer
adamLinear = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss);
adamLogistic = tf.train.AdamOptimizer(LEARNING_RATE).minimize(crossLoss);

#initialize global vars. need to do this to use variables
initializer = tf.global_variables_initializer();

with tf.Session() as sess:
    
    sess.run(initializer);
        
    numBatches = NUM_POINTS / BATCH_SIZE;
    start = 0;
    #descend the gradient 
    for i in range(NUM_ITERATIONS):
    
        start = (start+ BATCH_SIZE) % NUM_POINTS;
        end = start + BATCH_SIZE;
        sess.run(adamLinear, feed_dict={x:trainX[start : end]  , y:trainY[start : end] });      
        sess.run(adamLogistic,feed_dict={x:trainX[start : end]  , y:trainY[start : end] });    

        if( (i+ 1) % 7 == 0):
            [mseloss, crossError] = sess.run([loss, crossLoss], feed_dict={x:trainX[start : end]  , y:trainY[start : end] });
            mseValues.append(mseloss);
            crossValues.append(crossError)

            [mseClass, crossClass] = sess.run([mseClassError, crossClassError], feed_dict={x:trainX[start : end]  , y:trainY[start : end] });
            
            mseClassValues.append(mseClass);
            crossClassValues.append(1 - crossClass)



plt.title("Accuracy of Linear vs Logistic")
plt.xaxis("epochs")
plt.yaxis("accuracy")
plt.plot(mseClassValues, label="linear");
plt.plot(crossClassValues, label = "logistic");
plt.legend();
plt.show();
