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

bias = np.ones((3500,1), np.float64);
trainXBias = np.concatenate((bias,trainX), 1);


x = tf.placeholder(tf.float64, name="input_points_with_bias");
y = tf.placeholder(tf.float64, name="targets");
xReal = tf.placeholder(tf.float64, name="input_points");

#w = tf.matmul(tf.matrix_inverse( tf.matmul(tf.transpose(x), x) ), x)
wstar =tf.matmul(tf.matmul(tf.matrix_inverse( tf.matmul(tf.transpose(x), x) ), tf.transpose(x) ), y); 
b = wstar[0];
w = wstar[1:];

yhat = tf.add(tf.matmul(xReal,w), b); 
loss = tf.reduce_mean(((y - yhat)**2 )/2); 

start = time.time();
print(tf.Session().run(loss, feed_dict={x: trainXBias, y: trainY, xReal: trainX}) );

timeTaken = time.time() - start;
print("time taken: " + str(timeTaken));
