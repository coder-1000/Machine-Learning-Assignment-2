import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt



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
NUM_ITERATIONS = 200; #20000
LEARNING_RATE = [0.005, 0.001, 0.0001];
NUM_LR = 3;
BATCH_SIZE = 500;
NUM_BATCHES = 7;

trainX = np.reshape(trainData, (3500, PIXELS) );
trainY = trainTarget.astype(np.float64);

batchesX = np.array(np.split(trainX, NUM_BATCHES));
batchesY = np.array(np.split(trainY, NUM_BATCHES));

x = tf.placeholder(tf.float64, [BATCH_SIZE, PIXELS], name="input_points");
y = tf.placeholder(tf.float64, [BATCH_SIZE, 1], name="targets");

#initialize values to all 1
w = tf.Variable( np.ones((PIXELS,1), np.float64) ); #784 x 1 containing weights
b = tf.Variable(tf.constant(1,tf.float64), tf.float64); #offset


# (500x784) X (784x1) => (500x1) 
yhat = tf.add(tf.matmul(x,w), b); 

#loss_d = wx + b - y
#(500x1) => (1)
loss = tf.reduce_mean(tf.reduce_sum((y - yhat)**2, 1))/2  ; #check dimensions

lossValues = np.empty([NUM_LR, NUM_ITERATIONS], np.float64);

#initialize global vars. need to do this to use variables
initializer = tf.global_variables_initializer();

#print("this is the shape of trainX: ", trainX.shape);
#print("this is the shape of batchesX: ", batchesX.shape);
#print("this is the shape of batchesY: ", batchesY.shape);

#print("this is the shape of a batchX: ", batchesX[2].shape);
#print("this is the type of a batchX: ", batchesX.dtype);

#print("this is the shape of a batchY: ", batchesY.shape);
#print("this is the type of a batchY: ", batchesY.dtype);

with tf.Session() as sess:
   sess.run(initializer);  
   lrNum = 0;
   for learningRate in LEARNING_RATE:
        
        #reset the values for each learning rate so one doesn't have an unfair advantage over another
            
        print(sess.run(tf.reduce_mean(w) ) );    
        #set up optimizer
        sgdOptimizer = tf.train.GradientDescentOptimizer(learningRate).minimize(loss);
        
        #descend the gradient 
        for i in range(NUM_ITERATIONS*NUM_BATCHES):
            sess.run(sgdOptimizer, feed_dict={x: batchesX[i%NUM_BATCHES] , y: batchesY[i%NUM_BATCHES]});       
            if( i % NUM_BATCHES == 0):
                lossValues[lrNum][i/NUM_BATCHES] = sess.run(loss, feed_dict={x: batchesX[i%NUM_BATCHES] , y: batchesY[i%NUM_BATCHES]} );
            #if(i % 10000 == 0):    
                #print("iteration: ", i);
        
        print(sess.run(tf.reduce_mean(w) ) );        
        #print(sess.run(loss, feed_dict={x: batchesX[i%NUM_BATCHES] , y: batchesY[i%NUM_BATCHES]} ) )
        lrNum+=1;
