import tensorflow as tf

matrix1 = tf.constant([[3.,3.]])
matrix2 = tf.constant([[2.],[2.]])
#Q: 为什么数字后面有一个点，代表什么？
#A: 加点代表是浮点数float32类型，否则默认是int32类型

result = tf.matmul(matrix1, matrix2)

#launch default graph
sess = tf.Session()

final = sess.run(result)
print(final)
#预期输出[[ 12.]]

#close session when we finish
#不加close也可以正常运行，有什么影响？
sess.close()