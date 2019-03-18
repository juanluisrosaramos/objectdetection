import tensorflow as tf
import matplotlib.pyplot as plt

file_contents = tf.read_file('images/cjpf22efn42x20729xms4ql7n.jpeg')
im = tf.image.decode_jpeg(file_contents)
im_bi = tf.image.resize_images(im, 256, 256, method=tf.image.ResizeMethod.BILINEAR)
im_nn = tf.image.resize_images(im, 256, 256, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
im_bic = tf.image.resize_images(im, 256, 256, method=tf.image.ResizeMethod.BICUBIC)
im_ar = tf.image.resize_images(im, 256, 256, method=tf.image.ResizeMethod.AREA)

# im = tf.reshape(im, shape=[256, 256, 3])

sess = tf.Session()
sess.run(tf.initialize_all_variables())

img_bi, img_nn, img_bic, img_ar = sess.run([im_bi, im_nn, im_bic, im_ar])

plt.imshow(img_bi)
plt.title("BILINEAR")
plt.figure()

plt.imshow(img_nn)
plt.title("NEAREST_NEIGHBOR")
plt.figure()

plt.imshow(img_bic)
plt.title("BICUBIC")
plt.figure()

plt.imshow(img_ar)
plt.title('AREA')
plt.show()