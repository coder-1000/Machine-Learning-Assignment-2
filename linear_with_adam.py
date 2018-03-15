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
NUM_POINTS = 3500;

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

xValid = tf.placeholder(tf.float64, name="valid_inputs");
yValid =tf.placeholder(tf.float64, name="valid_y");

xTest = tf.placeholder(tf.float64, name="test_inputs");
yTest = tf.placeholder(tf.float64, name="test_y");

#w = tf.matmul(tf.matrix_inverse( tf.matmul(tf.transpose(x), x) ), x)
wstar =tf.matmul(tf.matmul(tf.matrix_inverse( tf.matmul(tf.transpose(x), x) ), tf.transpose(x) ), y); 
b = wstar[0];
w = wstar[1:];

yhat = tf.add(tf.matmul(xReal,w), b); 
yhatValid = tf.add(tf.matmul(xValid,w), b); 
yhatTest = tf.add(tf.matmul(xTest,w), b); 

classError = tf.reduce_mean( tf.abs( tf.round(yhat) - y) );

trainClassError = tf.reduce_mean( tf.abs( tf.round(yhat) - y) );
validClassError = tf.reduce_mean( tf.abs( tf.round(yhatValid) - yValid) );
testClassError =  tf.reduce_mean( tf.abs( tf.round(yhatTest) - yTest) );

[train, valid, test] = tf.Session().run(
    [trainClassError, validClassError, testClassError], 
    feed_dict= {
        x: trainXBias, xReal: trainX, y: trainY, 
        xValid: validX, yValid: validY, 
        xTest: testX, yTest: testY
    }
);

print("training class error  : " + str(train));
print("validation class error: " + str(valid));
print("test class error      : " + str(test));





