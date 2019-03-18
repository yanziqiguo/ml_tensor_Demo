import tensorflow as tf

zeros = tf.zeros((2,3), dtype=tf.int32)

sess = tf.Session()

print('{}:\n{}\n'.format('zeros', sess.run(zeros)))


sess.close()