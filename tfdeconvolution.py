# http://blog.simiacryptus.com/2016/01/deblurring-with-tensorflow.html

import tensorflow as tf
import mahotas
import skimage.io as imgio
import numpy
import scipy.misc

if __name__ == '__main__':
    p_imgfile = 'monkey-02.jpg'
    p_imgscale = .33
    p_ksize = 4
    p_kiter = 3
    learning_rate = 0.1

    # preprocess the image: resizing and conversion to float
    img_raw = mahotas.imread(p_imgfile)
    img_base = scipy.misc.imresize(img_raw, p_imgscale) / 255.0
    imgio.imsave('base.png', img_base)

    # building the blur kernel
    kernel = numpy.zeros([p_ksize, p_ksize, 3, 3])  # 3 channels in -> 3 channels out
    for c in range(0, 3):
        for xy in range(0, p_ksize):
            kernel[xy, xy, c, c] = 1.0 / p_ksize

    # the variable will be tuned by the optimizer to contain the unblurred image given
    # the known blurring kernel
    v_img = tf.Variable(tf.zeros(img_base.shape), name="Unblurred_Image")

    # build a subgraph to blur the image
    op_img_resize = tf.reshape(v_img, [-1, img_base.shape[0], img_base.shape[1], img_base.shape[2]])
    pl_kernel = tf.placeholder("float", shape=kernel.shape, name="Kernel")
    op_init = tf.global_variables_initializer()

    # blur the image repeatedly
    op_convolve = op_img_resize
    for blurStage in range(0, p_kiter):
        op_convolve = tf.nn.conv2d(op_convolve, pl_kernel, strides=[1, 1, 1, 1], padding='SAME')

    # the actual blurring operation
    with tf.Session() as session:
        session.run(op_init)
        img_blurred = session.run(op_convolve, feed_dict={v_img: img_base, pl_kernel: kernel})
    imgio.imsave('blurred.png', img_blurred[0])

    # build a subgraph to determine the estimation error and optimize on it
    pl_blurredImg = tf.placeholder("float", shape=img_blurred.shape)
    op_loss = tf.reduce_sum(tf.square(op_convolve - pl_blurredImg))

    op_optimize = tf.train.AdamOptimizer(learning_rate).minimize(op_loss)
    op_init = tf.global_variables_initializer()

    def f_pixel(x):
        return 0 if x < 0 else 1 if x > 1 else x

    f_img = numpy.vectorize(f_pixel, otypes=[numpy.float])

    # run the actual optimization
    with tf.Session() as session:
        session.run(op_init)

        for epoch in range(0, 5):
            # obtain the deblurred image
            img_deblurred = session.run(v_img, feed_dict={pl_blurredImg: img_blurred, pl_kernel: kernel})
            img_deblurred = f_img(img_deblurred)
            imgio.imsave("deblurred-%s.png" % epoch, img_deblurred)

            for iteration in range(0, 100):
                _, error = session.run([op_optimize, op_loss], feed_dict={pl_blurredImg: img_blurred, pl_kernel: kernel})
                print("%s/%s = %s" % (epoch, iteration, error))

        # obtain the deblurred image
        img_deblurred = session.run(v_img, feed_dict={pl_blurredImg: img_blurred, pl_kernel: kernel})
        img_deblurred = f_img(img_deblurred)
        imgio.imsave("deblurred-final.png", img_deblurred)
