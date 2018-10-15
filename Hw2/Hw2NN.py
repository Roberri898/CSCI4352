import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler

xy = np.loadtxt('iris.csv', delimiter=',', dtype = np.float32)


n_class = 3
n_train_samples=len(xy)

X = tf.placeholder(tf.float32, shape=[None, 4])
Y = tf.placeholder(tf.int32, shape=[None, 1])
Y_onehot = tf.one_hot(Y, 3)
Y_onehot = tf.reshape(Y_onehot, [-1, 3])

W1= tf.Variable(tf.random_normal([4,8]), name='weigth')
b1 = tf.Variable(tf.random_normal([8]), name='bias')
layer1 = tf.sigmoid(tf.matmul(X,W1) + b1)

W2 = tf.Variable(tf.random_normal([8,n_class]), name='weight')
b2 = tf.Variable(tf.random_normal([n_class]), name='bias')

hypothesis = tf.nn.softmax(tf.matmul(layer1 , W2) +b2)

cross_entropy = -tf.reduce_sum(Y_onehot*tf.log(hypothesis))
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
is_correct = tf.equal(tf.argmax(hypothesis,1), tf.argmax(Y_onehot,1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

n_epoch = 50
batch_size = 100

accuracyT=0
accuracyI=0

k=10
foldsize = int(len(xy)/k)  # 150/10 = 15
runs=0
sess= tf.Session()
sess.run(tf.global_variables_initializer())
for j in range(10):
	for i in range(k):
		np.random.shuffle(xy)
		a=i*foldsize
		b=(i+1)*foldsize-1
		traind = np.delete(xy, np.s_[a+1:b], axis=0)
		trainX = traind[:,:-1]
		trainY = traind[:,[-1]]


		scaler = MinMaxScaler()
		scaler.fit(trainX)
		trainX = scaler.transform(trainX)

		testX = xy[a:b,:-1]
		testY = xy[a:b,[-1]]
		scaler = MinMaxScaler()
		scaler.fit(testX)
		testX= scaler.transform(testX)


		for epoc in range(n_epoch):
			avg_cost = 0
			total_batch = int(n_train_samples/batch_size)
			for i in range(total_batch):
				c, _ =sess.run([cost,optimizer], feed_dict={X: trainX, Y: trainY})
				avg_cost += c/total_batch
			runs+=1
			print("Epoch:",'%04d' % (runs), 'cost=', avg_cost)
			accuracyT+=sess.run([accuracy], feed_dict={X: testX, Y: testY})[0]
			accuracyI+=1
			print("Avg_Accuracy:", accuracyT/accuracyI)
			print()
