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
BATCH_SIZE = 500;
NUM_POINTS = 3500;

LAMDA = [0., 0.001, 0.1, 1.];

trainX = np.reshape(trainData, (3500, -1) );
trainY = trainTarget.astype(np.float64);

validX = np.reshape(validData, (100, -1) );
validY = validTarget.astype(np.float64);

testX = np.reshape(testData, (145, -1) );
testY = testTarget.astype(np.float64);




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

for lamda  in LAMDA:
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
    loss = tf.reduce_mean(((y - yhat)**2 )/2) + tf.reduce_sum(w**2)*(lamda/2) ; #check dimensions
    class_error = tf.reduce_mean( tf.abs( tf.round(yhat) - y) );

    #initialize global vars. need to do this to use variables
    initializer = tf.global_variables_initializer();
    

    with tf.Session() as sess:
        sess.run(initializer);  
        
        #print('sanity checks 1');
        #print(sess.run(b, feed_dict={x: batchesX[0] , y: batchesY[0]}));
        #print(sess.run(tf.reduce_sum(w),feed_dict={x: batchesX[0] , y: batchesY[0]}));
        #print(sess.run(tf.reduce_mean(w),feed_dict={x: batchesX[0] , y: batchesY[0]}));
            
        #set up optimizer
        sgdOptimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss);
        

        start = 0;
        end = start + BATCH_SIZE;
        #descend the gradient 
        for i in range(NUM_ITERATIONS):
    
           # start = (start+ BATCH_SIZE) % NUM_POINTS;
          #  end = start + BATCH_SIZE;

            sess.run(sgdOptimizer, feed_dict={x:trainX[start : end]  , y:trainY[start : end] });
            start = (start + BATCH_SIZE) % NUM_POINTS;
            end = start + BATCH_SIZE;
   
        #print('sanity checks 2');
        #print(sess.run(b, feed_dict={x: batchesX[0] , y: batchesY[0]}));
        #print(sess.run(tf.reduce_sum(w),feed_dict={x: batchesX[0] , y: batchesY[0]}));
        #print(sess.run(tf.reduce_mean(w),feed_dict={x: batchesX[0] , y: batchesY[0]}));
        
        print("lambda is " + str(lamda) );

        testError = sess.run(class_error, feed_dict={x: testX, y: testY} ); 
        print('test error is ' + str(testError) + '\n');
        
        validationError = sess.run(class_error, feed_dict={x: validX, y: validY} );
        print('validation error is ' + str(validationError) + '\n');

        validLoss = sess.run(loss, feed_dict={x: trainX, y: trainY} );
        print('valid loss is ' + str(validLoss));
