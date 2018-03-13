#implement logistic regression 
#logistic regression prediction function is y = sigmoid(W^Tx + b)
#train the logistic regression model using SGD and mini batch size B = 500 on the two-class notNMIST dataset
#how to train the dataset:

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
##############Constants##################################

BATCH_SIZE = 500;
NUM_BATCHES = 7;
NUM_ITERATIONS = 5000;
LEARNING_RATE = [0.005]; #[0.005 0.001, 0.0001];
PIXEL_SIZE = 784; #28x28
NUM_TRAINING_POINTS = 3500;
NUM_VALID_POINTS = 100;
NUM_TEST_POINTS = 145;

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
validX = np.reshape(validData, (NUM_VALID_POINTS, PIXEL_SIZE))
testX = np.reshape(testData, (NUM_TEST_POINTS, PIXEL_SIZE))
batchesX = np.array(np.split(trainX, NUM_BATCHES));
batchesY = np.array(np.split(trainTarget, NUM_BATCHES));


################Defining variables########################

loss_Values = [[0 for x in range(NUM_BATCHES)] for y in range(715)]
lr = dict()
epoch_list = []
mean_list = []
accuracy_list = []
loss = []

x = tf.placeholder(tf.float32, [PIXEL_SIZE, None], name = "input_points") #784 dimensions (28x28 pixels) 
W = tf.Variable(tf.truncated_normal(shape=[PIXEL_SIZE,1], stddev=0.5), name='weights')
b = tf.Variable(0.0, name='bias')
y = tf.placeholder(tf.float32, [None,1], name = "target_labels")#target labels
lambda_ = 0.01

##############Calculations###############################

#weight_squared_sum = tf.matmul(tf.transpose(W),W) #find the square of the weight vector
#calculating the bias term

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    weight = W.eval()

weight_squared_sum = np.linalg.norm(weight)
loss_W = lambda_ /2 * weight_squared_sum #find the loss

y_hat = tf.add(tf.matmul(tf.transpose(W), x), b) #based on the sigmoid equation given in lab handout (logits)
y_hat = tf.transpose(y_hat)
cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits = y_hat, labels = y) #sigmoid_cross_entropy_with_logits takes in the actual y and the predicted y 
total_loss = tf.add(tf.reduce_mean(cross_entropy,0),loss_W) #does it matter if this is 1 or 0?

#############Training######################################
epoch = 0

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    
    for learning_rate in LEARNING_RATE:
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss) #change the learning rate each time
        for i in range(NUM_BATCHES*NUM_ITERATIONS):  
            #at each batch, store the loss and average over the loss vector
            temp = sess.run(cross_entropy, feed_dict={x:np.transpose(batchesX[i%NUM_BATCHES]), y: batchesY[i%NUM_BATCHES]})
            temp = np.mean(temp)
            loss.append(temp)    

            sess.run(train_step, feed_dict={x:np.transpose(batchesX[i%NUM_BATCHES]), y: batchesY[i%NUM_BATCHES]})
            
            if( i % NUM_BATCHES == 0): #everytime we reach 0, a new batch has started and therefore 1 epoch has been completed
                #average over all the losses in the epoch
                loss_val = np.mean(loss)                
                loss_Values[epoch]= loss_val;
        
                #find prediction and compare with actual value to find accuracy
                y_hat_new = tf.sigmoid(y_hat)
                y_hat_new = sess.run(y_hat_new,feed_dict={x: np.transpose(validX), y: validTarget})
                print("Epoch:", epoch)
                print("Learning Rate:", learning_rate)

				#print("i", i)
                #print("Y_HAT:")
                y_hat_new = np.around(y_hat_new)
                #print(y_hat_new)
                #print("Y:")
                y_new = sess.run(y,feed_dict = {y: validTarget})
                y_new = np.around(y_new)
                #print(y_new)
                correct_prediction = np.equal(y_new, y_hat_new)
                accuracy = np.mean(correct_prediction)
                
                print("ACCURACY:")
                print(accuracy)
                print("LOSS_VALUES")
                print(loss_val)
        
                accuracy_list.append(accuracy) 
                #print("Accuracy_list", accuracy_list)
                epoch = epoch + 1;
                
                #reset the loss list
                loss = []
                if(epoch == 715):
					break;



    #for plotting
	epoch_list = [x for x in range(715)]
    epoch_list = np.array(epoch_list)
    mean_list = np.array(mean_list)
    accuracy_list = np.array(accuracy_list)
    print("Optimization complete!")
    print("Accuracy list")
    print(accuracy_list)

    plt.figure()
    plt.plot(epoch_list, accuracy_list, '-')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Epochs vs Accuracy')
    plt.show()

    plt.figure()
    plt.plot(epoch_list, loss_Values, '-', label = 'Epoch vs Average Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Entropy Loss')
    plt.title('Epochs vs Entropy Loss')
    plt.show()
    plt.show()


	
##########################################################
