import tensorflow as tf
sess = tf.InteractiveSession()

x = tf.Variable([1, 2])
y = tf.constant([3, 3])

#Q: 为什么要初始化操作 x变量？
#A: 变量必须经过初始化才能使用，否则会提示尝试使用未初始化的变量
x.initializer.run()

#subtract 减法
#eval 计算  ==>  t.eval()
#Q: 不使用tf.Session.run()可以避免使用一个变量来"持有"会话?
#A: ???
print(tf.subtract(x, y).eval())
#预期结果
#[-2 -1]

sess.close()
