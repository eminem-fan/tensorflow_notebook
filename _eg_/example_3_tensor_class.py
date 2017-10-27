import tensorflow as tf

matrix1 = tf.constant([[1],[2]])
matrix2 = tf.constant([[3, 4]])

#矩乘
result = tf.matmul(matrix1, matrix2)

with tf.Session() as sess:
	final = sess.run(result)
	print("matrix1:", matrix1)
	print("matrix1 after sess.run()):\n", sess.run(matrix1))
	print("matrix1 after t.eval():\n", matrix1.eval())
	print("matrix2:", matrix2)
	print("matrix2 after sess.run:\n", sess.run(matrix2))
	print("result:", result)
	print("***********")
	print(matrix1.consumers())
	print(matrix2.consumers())
	print(result.consumers())
	print("***********")
	print(final)

#预期结果：
#matrix1: Tensor("Const:0", shape=(2, 1), dtype=int32)
#matrix1 after sess.run()):
# [[1]
# [2]]
#matrix1 after t.eval():
# [[1]
# [2]]
#matrix2: Tensor("Const_1:0", shape=(1, 2), dtype=int32)
#matrix2 after sess.run:
# [[3 4]]
#result: Tensor("MatMul:0", shape=(2, 2), dtype=int32)
#***********
#[<tf.Operation 'MatMul' type=MatMul>]
#[<tf.Operation 'MatMul' type=MatMul>]
#[]
#***********
#[[3 4]
# [6 8]]