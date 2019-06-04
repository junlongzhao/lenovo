import tensorflow as tf
with tf.Session() as sess:
  a=tf.ones(shape=(2,3))
  b=tf.ones(shape=(2,3))
  print(sess.run(a+b))