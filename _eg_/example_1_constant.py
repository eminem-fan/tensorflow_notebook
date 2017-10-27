import tensorflow as tf

blk1 = tf.constant([[5.], [6.]])
blk2 = tf.constant([[7., 8.]])
print(blk1)
#matmul即线代里的矩阵乘法
#e.g. C = matmul(A, B), A的第一行和B的第一列分别相乘数据存放，A按行向下递进，B向右递进。
cross = tf.matmul(blk1, blk2)

#利用with调用Session方法，会话会在with句块结束后自动关闭，省略sess.close操作
#tf只有Session对象，不要写错成session
with tf.Session() as sess:
	result = sess.run(cross)
	print(result)

#预期结果：
#[[ 35.  40.]
# [ 42.  48.]]