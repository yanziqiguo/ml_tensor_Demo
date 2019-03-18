import tensorflow as tf

a = tf.constant(32)
b=tf.constant(10)
c=tf.add(a,b)
print(a)
print(b)
print(c)

sess = tf.Session()
print(sess.run(a))
print(sess.run([a,b]))
print(sess.run([a,b,c]))

# fetch the data
py_a = sess.run(a)
print(type(py_a))
print(py_a)

py_r = sess.run([a,b,c])
print(type(py_r))
print(py_r[0], py_r[1], py_r[2])

hello = tf.constant("Hello, TensorFlow")
boolen = tf.constant(True)
int_array = tf.constant([1,2], dtype=tf.int32)
float_array = tf.constant([1,2],dtype=tf.float32)

print(sess.run(hello))
print(sess.run(boolen))
print(sess.run(int_array))
print(sess.run(float_array))

mat = tf.constant([[1,0],[0,1]])
print(sess.run(mat))


my_name_hello = tf.constant("Hello", name='Hello')
my_name_world = tf.constant("World", name="World")
print("tensor {}: {}".format(my_name_hello.name, sess.run(my_name_hello)))
print("tensor {}: {}".format(my_name_world.name, sess.run(my_name_world)))

d = tf.add_n([a,b,c])
e = tf.subtract(a,b)
f = tf.multiply(a,b)
g = tf.divide(a,b)
h = tf.mod(a,b)

print(sess.run(d))
print(sess.run(e))
print(sess.run(f))
print(sess.run(g))
print(sess.run(h))

a_float = tf.cast(a,dtype=tf.float32)
b_float = tf.cast(b,dtype=tf.float32)

i = tf.sin(a_float)
j = tf.exp(tf.divide(1.0,a_float))
k = tf.add(i, tf.log(i))


print(sess.run(i))
print(sess.run(j))
print(sess.run(k))


sigmoid = tf.divide(1.0, tf.add(1.0, tf.exp(-b_float)))
print(sess.run(sigmoid))

mat_a = tf.constant([1,2,3,4])
mat_a = tf.reshape(mat_a, (2,2))
mat_b = tf.constant([1,3,5,7,9,11])
mat_b = tf.reshape(mat_b, (2,3))
vec_a = tf.constant([1,2])

mat_c = tf.matmul(mat_a, mat_b)
mat_d = tf.multiply(mat_a, vec_a)
print(sess.run(mat_a))
print(sess.run(mat_b))
print(sess.run(mat_c))
print(sess.run(mat_d))


sess.close()



