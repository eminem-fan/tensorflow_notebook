#from __future__ import print_function,division
import numpy as np
import tensorflow as tf

#create a Variable
w=tf.Variable(initial_value=[[1,2],[3,4]],dtype=tf.float32)
x=tf.Variable(initial_value=[[1,1],[1,1]],dtype=tf.float32)
y=tf.matmul(w,x)
#sigmoid常用于神经网络中常用的阈值函数，把变量映射到0到1之间，公式为y = 1/(1+e-x)，引入非线性
#导数形式: S(x)(1-S(x))
z=tf.sigmoid(y)
print(w)
print(x)
print(y)
print(z)
#初始化所有全局变量？
init_op=tf.global_variables_initializer()

#<tf.Variable 'Variable:0' shape=(2, 2) dtype=float32_ref>
#<tf.Variable 'Variable_1:0' shape=(2, 2) dtype=float32_ref>
#Tensor("MatMul:0", shape=(2, 2), dtype=float32)
#Tensor("Sigmoid:0", shape=(2, 2), dtype=float32)

with tf.Session() as sess:
#	w.initializer.run()
#	x.initializer.run()
	sess.run(init_op)
	#w1 = sess.run(w)
	y1 = sess.run(y)
	print("*****temp*****")
	#print(w1)
	print(w.eval())
	print(x.eval())
	print(y1)

#*****temp*****
#[[ 1.  2.]
# [ 3.  4.]]
#[[ 1.  1.]
# [ 1.  1.]]
#[[ 3.  3.]
# [ 7.  7.]]
	
with tf.Session() as session:
	session.run(init_op)
	z=session.run(z)
	print("**********")
	print(z)

#**********
#[[ 0.95257413  0.95257413]
# [ 0.999089    0.999089  ]]