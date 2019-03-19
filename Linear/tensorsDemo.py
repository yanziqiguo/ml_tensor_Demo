import tensorflow as tf


sess = tf.Session()

zeros = tf.zeros((2,3), dtype=tf.int32)
zeros_like = tf.zeros_like(zeros)
ones_like = tf.ones_like(zeros)
fill = tf.fill((2,4), 2)
linespace = tf.linspace(1.0, 5.0, 5)
ranger = tf.range(3,8,delta=3)

print('{}:\n{}\n'.format('zeros', sess.run(zeros)))
print('{}:\n{}\n'.format('zeros_like', sess.run(zeros_like)))
print('{}:\n{}\n'.format('ones_like', sess.run(ones_like)))
print('{}:\n{}\n'.format('fill', sess.run(fill)))
print('{}:\n{}\n'.format('linespace', sess.run(linespace)))
print('{}:\n{}\n'.format('ranger', sess.run(ranger)))


#=================
rand_normal = tf.random_normal((), mean=0.0,stddev=1.0,dtype=tf.float32, seed=None)
truncated_normal = tf.truncated_normal((), mean=0.0, stddev=1.0, dtype=tf.float32,seed=None)
rand_uniform = tf.random_uniform((), minval=0.0, maxval=1.0, dtype=tf.float32,seed=None)
for i in range(5):
    print('time: %d' % i)
    print('rand_normal: %.4f' % sess.run(rand_normal))
    print('truncated_normal: %.4f' % sess.run(truncated_normal))
    print('rand_uniform: %.4f' % sess.run(rand_uniform))

hello = tf.constant("Hello, TensorFlow")
print(sess.run(hello))

sess.close()


#===================
var_a = tf.Variable(0, dtype=tf.float32)
var_b = tf.Variable([1,2], dtype=tf.float32)
var_w = tf.Variable(tf.zeros((1024, 10)))

session = tf.InteractiveSession()

#init all variable
init = tf.global_variables_initializer()

#general variable
session.run(init)
#interactiveSession method
init.run()

init_ab = tf.variables_initializer([var_a,var_b])
init_ab.run()

var_w.initializer.run()

W = tf.Variable(10)
session.run(W.initializer)
print(W)
print(session.run(W))
print(W.eval())

assign_op = W.assign(100)
W.initializer.run()
assign_op.eval()
print(W.eval())


assign_add = W.assign_add(10)
assign_sub = W.assign_sub(2)
W.initializer.run()
print(assign_add.eval())
print(assign_sub.eval())
print(W.eval())

with tf.name_scope('name_scope'):
    var_a = tf.Variable(0, dtype=tf.int32)
    var_b = tf.Variable([1,2], dtype=tf.float32)

with tf.variable_scope('var_scope'):
    var_c = tf.Variable(0, dtype=tf.int32)
    var_d = tf.Variable([1,2], dtype=tf.float32)
print(var_a.name)
print(var_b.name)
print(var_c.name)
print(var_d.name)

a = tf.placeholder(tf.float32,shape=[3])

b = tf.placeholder(tf.bool, shape=[1,2])

print(session.run(a, feed_dict={a:[1,2,3]}))
print(session.run([a,b],feed_dict={a:[1,2,3], b:[[True,False]] }))

c = tf.placeholder(tf.float32)
square = tf.square(c)
for i in [1,2,4,8]:
    print(square.eval(feed_dict={c:i}))

g= tf.get_default_graph()
print(g)

for op in g.get_operations():
    print(op.name)

# what_is_this = g.get_tensor_by_name('Hello, TensorFlow')
# print(what_is_this.eval())

g1 = tf.Graph()
print('g1:', g1)
print('default_graph:', tf.get_default_graph())
g1.as_default()
print('default_graph:', tf.get_default_graph())

a1 = tf.constant(32,name='a1')
with g1.as_default():
    a2 = tf.constant(32, name='a2')

print('a.graph: ', a.graph)
print('a1.graph: ', a1.graph)
print('a2.graph: ', a2.graph)


with tf.Session() as sess:
    graph_writer = tf.summary.FileWriter('.', sess.graph)




session.close()





