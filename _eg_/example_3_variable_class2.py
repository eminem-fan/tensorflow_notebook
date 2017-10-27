#get_shape(), read_value(), load(), eval(), eval()

import tensorflow as tf

a = tf.Variable([2,3])
print(a)
print("******")
c = a.read_value()
#result1 = a.assign_add([2,3], use_locking=True)
#result2 = result1.assign_sub([1,1], use_locking=True)

#tf.global_variables_initializer()

with tf.Session() as sess:
	a.initializer.run()
	#a.load([88,99])
	print(a.get_shape())
	print(a.initialized_value())	#返回初始变量值
	print(a.eval())
	print(a.assign_add([1,1], use_locking=True))
	print(a.eval())
#	print(a.assign_sub(2, use_locking=True))
#	print(a.eval())
	print(c.eval())
#b = a.scatter_sub([1,2], use_locking = True)
#with tf.Session() as sess:
#	result = sess.run(b)
#	print(result)

#	b.read_value()