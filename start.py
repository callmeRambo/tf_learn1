import tensorflow as tf
import numpy as np
#initialize x and y there
#2维，每维1oo个数字
x_data = np.float32(np.random.rand(2,100))
#print(x_data)
#点乘
y_data = np.dot([0.100,0.200],x_data)+0.300
#定义变量
b = tf.Variable(tf.zeros([1]))
#一行两列
W = tf.Variable(tf.random_uniform([1,2],-1.0,1.0))
y = tf.matmul(W,x_data) + b

loss = tf.reduce_mean(tf.square(y-y_data))
#学习效率0.5
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for step in range(0,201):
    sess.run(train)
    if step%20 == 0:
        print(step,sess.run(W),sess.run(b))