

#implement logistic regression 
#logistic regression prediction function is y = sigmoid(W^Tx + b)
#train the logistic regression model using SGD and mini batch size B = 500 on the two-class notNMIST dataset
#how to train the dataset:

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

##############Constants##################################

BATCH_SIZE = 500;
NUM_BATCHES = 7;
NUM_ITERATIONS = 5000;
LEARNING_RATE = [0.005]#, 0.001, 0.0001];
PIXEL_SIZE = 784; #28x28
NUM_TRAINING_POINTS = 3500;

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

################Manipulating Data##########################

trainX = np.reshape(trainData, (NUM_TRAINING_POINTS, PIXEL_SIZE));
batchesX = np.array(np.split(trainX, NUM_BATCHES));
batchesY = np.array(np.split(trainTarget, NUM_BATCHES));
print("BATCHESY")
print(batchesY.shape)

################Defining variables########################

loss_Values = [[0 for x in range(NUM_BATCHES)] for y in range(715)]
epoch_list = []
mean_list = []

#W = tf.Variable(tf.zeros([PIXEL_SIZE,1]), name = "weights") #weight vector
#b = tf.Variable(tf.constant(1,tf.float32), tf.float32, name = "bias"); #offset/bias
#lambda_ = tf.placeholder(tf.float32) #constant
x = tf.placeholder(tf.float32, [PIXEL_SIZE, None], name = "input_points") #784 dimensions (28x28 pixels) 
W = tf.Variable(tf.truncated_normal(shape=[PIXEL_SIZE,1], stddev=0.5), name='weights')
b = tf.Variable(0.0, name='bias')
y = tf.placeholder(tf.float32, [None,1], name = "target_labels")#target labels
lambda_ = 0.01
#learning_rate = tf.placeholder(tf.float32) #constant

##############Calculations###############################

#weight_squared_sum = tf.matmul(tf.transpose(W),W) #find the square of the weight vector
#calculating the bias term

with tf.Session() as sess:
	tf.global_variables_initializer().run()
	weight = W.eval()

weight_squared_sum = np.linalg.norm(weight)
loss_W = lambda_ /2 * weight_squared_sum #find the loss

#do I need to broadcast the b or is that taken care of?
#print("SHAPE:")
#print(x.shape)
#print(W.shape)
#print(tf.matmul(tf.transpose(W),x).shape)

y_hat = tf.add(tf.matmul(tf.transpose(W), x), b) #based on the sigmoid equation given in lab handout (logits)
y_hat = tf.transpose(y_hat)
#print("Y_HAT:")
#print(y_hat.shape)#this is (1,?)

#print("Y-shape")
#print(y.shape)

cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits = y_hat, labels = y) #sigmoid_cross_entropy_with_logits takes in the actual y and the predicted y 
#print("CROSS_ENTROPY:")
#print(cross_entropy.shape) #<unknown>

#print("REDUCE_MEAN")
#print(tf.reduce_mean(cross_entropy,0).shape)

#print("LOSS_W")
#print(loss_W.shape)

total_loss = tf.add(tf.reduce_mean(cross_entropy,0),loss_W) #does it matter if this is 1 or 0?
#print("TOTAL_LOSS:")
#print(total_loss.shape) # <unknown>


#print("TRAIN_STEP:")
#print(train_step.shape) #error: has no shape

#############Training######################################

with tf.Session() as sess:
	epoch = 0;
	tf.global_variables_initializer().run()
    
	for learning_rate in LEARNING_RATE:
		print(total_loss)
		print(learning_rate)

		train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss) #change the learning rate each time
		print("TRAIN_STEP")
		#print(train_step.shape)

		for i in range(NUM_ITERATIONS*NUM_BATCHES):	
			sess.run(train_step, feed_dict={x:np.transpose(batchesX[i%NUM_BATCHES]), y: batchesY[i%NUM_BATCHES]})
			print(i)
			print(sess.run(total_loss, feed_dict={x:np.transpose(batchesX[i%NUM_BATCHES]), y: batchesY[i%NUM_BATCHES]}))

			if( i % NUM_BATCHES == 0): #everytime we reach 0, a new batch has started and therefore 1 epoch has been completed
				epoch = i%NUM_BATCHES%NUM_BATCHES;
				loss_Values[epoch][i%NUM_BATCHES] = sess.run(cross_entropy, feed_dict={x: np.transpose(batchesX[i%NUM_BATCHES]) , y: batchesY[i%NUM_BATCHES]});
		    	epoch = epoch + 1;
    
	print("Final value")
	print(sess.run(tf.reduce_mean(W)))
	
	#for plotting purposes             
	N = len(loss_Values)
	for epoch in range (N):
		epoch_list.append(epoch)
		row = np.array(loss_Values[epoch])
		mean = np.mean(row)
		mean_list.append(mean)

	
	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_hat,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	print(sess.run(accuracy, feed_dict={x:np.reshape(np.transpose(testData),(784,145)), y:testTarget}))
	
	epoch_list = np.array(epoch_list)
	mean_list = np.array(epoch_list)

	plt.figure()
	plt.plot(epoch_list, mean_list, '-', label = 'Average loss')
	plt.show()

	#plt.figure()
	#plt.plot(epoch_list, accuracy, '.', label = "Accuracy vs epochs")
	#plt.show()

##########################################################
