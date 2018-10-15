import tensorflow as tf
import numpy as np

xy = np.loadtxt('iris.csv', delimiter=',', dtype=np.float32)
X = tf.placeholder(tf.float32, shape=[None,4])
Y = tf.placeholder(tf.int32, shape=[None,1])
Y_onehot = tf.one_hot(Y, 3)
Y_onehot = tf.reshape(Y_onehot, [-1, 3])
W = tf.Variable(tf.random_normal([4,3]))
b = tf.Variable(tf.random_normal([3]))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
hypothesis =tf.matmul(X, W) + b
cost = tf.reduce_mean(tf.square(hypothesis-Y_onehot))
train = optimizer.minimize(cost)

accuracyT=0
accuracyI=0
k=10
foldsize= int(len(xy)/k)
for j in range(10):
	for i in range(k):
		np.random.shuffle(xy)
		a = i*foldsize
		b = (i+1)*foldsize-1
		traind = np.delete(xy, np.s_[a+1:b], axis=0)
		dataX = traind[:,:-1]
		dataY = traind[:,[-1]]
		testX = xy[a:b,:-1]
		testY = xy[a:b,[-1]]
		sess = tf.Session()
		sess.run(tf.global_variables_initializer())
		for step in range(20000):
			sess.run(train, feed_dict={X:dataX, Y: dataY})
			if step % 5000 == 0:
				 print(step, sess.run(cost, feed_dict={X: dataX, Y: dataY}))
		is_correct = tf.equal(tf.argmax(hypothesis,1), tf.argmax(Y_onehot, 1))
		accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
		accuracyT+=sess.run([accuracy], feed_dict={X: testX, Y: testY})[0]
		accuracyI+=1
		print('Accuracy {}'.format(accuracyT/accuracyI))
		print(sum(np.argmax(sess.run(hypothesis, feed_dict={X:testX}), axis=1)==np.argmax(testY,axis=1))/len(testY))
