import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


yhat_val = np.reshape(np.linspace(0,1,100),(100,1))
y_val = np.reshape(np.zeros(100),(100,1))


y = tf.placeholder(tf.float32, [None,1], name = "target_labels")#target labels
yhat = tf.placeholder(tf.float32, [None,1], name = "prediction_labels")#target labels

cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits = yhat, labels = y)
loss = ((y - yhat)**2)/2  ; #check dimensions

with tf.Session() as sess:
    entropy_loss = sess.run(cross_entropy, feed_dict = {yhat:yhat_val, y:y_val})
    mse_loss = sess.run(loss, feed_dict = {yhat:yhat_val, y:y_val})


plt.figure()
plt.plot(yhat_val, entropy_loss, '-', label = 'Entropy Loss')
plt.plot(yhat_val, mse_loss, '-', label = 'MSE Loss')
plt.xlabel('Predictions')
plt.ylabel('Loss')
plt.legend()

plt.show()
