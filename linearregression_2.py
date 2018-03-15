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
LEARNING_RATE = 0.005;#best learning rate from pt.1 
BATCH_SIZE = [500,1500,3500];
NUM_POINTS = 3500;

trainX = np.reshape(trainData, (3500, -1) );
trainY = trainTarget.astype(np.float64);

#making batches was stupid
#batchesX = np.array(np.split(trainX, NUM_BATCHES));
#batchesY = np.array(np.split(trainY, NUM_BATCHES));


#print("this is the shape of trainX: ", trainX.shape);
#print("this is the shape of batchesX: ", batchesX.shape);
#print("this is the shape of batchesY: ", batchesY.shape);

#print("this is the shape of a batchX: ", batchesX[2].shape);
#print("this is the type of a batchX: ", batchesX.dtype);

#print("this is the shape of a batchY: ", batchesY.shape);
#print("this is the type of a batchY: ", batchesY.dtype);


lrNum = 0;

for batchSize  in BATCH_SIZE:
    #reset w and b
    tf.reset_default_graph();

    x = tf.placeholder(tf.float64, name="input_points");
    y = tf.placeholder(tf.float64, name="targets");

    #initialize values to all 1
    w = tf.Variable( np.zeros((PIXELS,1), np.float64) ); #784 x 1 containing weights
    b = tf.Variable(tf.constant(0,tf.float64), tf.float64); #offset


    # (500x784) X (784x1) => (500x1) 
    yhat = tf.add(tf.matmul(x,w), b); 

    #loss_d = wx + b - y
    #(500x1) => (1)
    loss = tf.reduce_mean((y - yhat)**2)/2  ; #check dimensions


    #initialize global vars. need to do this to use variables
    initializer = tf.global_variables_initializer();
    
    startTime = time.time();

    with tf.Session() as sess:
        sess.run(initializer);  
        
        #print('sanity checks 1');
        #print(sess.run(b, feed_dict={x: batchesX[0] , y: batchesY[0]}));
        #print(sess.run(tf.reduce_sum(w),feed_dict={x: batchesX[0] , y: batchesY[0]}));
        #print(sess.run(tf.reduce_mean(w),feed_dict={x: batchesX[0] , y: batchesY[0]}));
            
        #set up optimizer
        sgdOptimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss);
        
        numBatches = NUM_POINTS / batchSize;
        start = 0;
        #descend the gradient 
        for i in range(NUM_ITERATIONS):
    
            start = (start+ batchSize) % NUM_POINTS;
            end = start + batchSize;

            sess.run(sgdOptimizer, feed_dict={x:trainX[start : end]  , y:trainY[start : end] });       
   
        #print('sanity checks 2');
        #print(sess.run(b, feed_dict={x: batchesX[0] , y: batchesY[0]}));
        #print(sess.run(tf.reduce_sum(w),feed_dict={x: batchesX[0] , y: batchesY[0]}));
        #print(sess.run(tf.reduce_mean(w),feed_dict={x: batchesX[0] , y: batchesY[0]}));
     
        timeTaken = time.time() - startTime; 
        print('time taken for batch size ' + str(batchSize) + ' is ' + str(timeTaken) );
        lossOverData = sess.run(loss, feed_dict={x: trainX, y: trainY} );
        print('loss is ' + str(lossOverData) + '\n');
