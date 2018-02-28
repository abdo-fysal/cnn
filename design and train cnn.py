import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
p=cv2.imread('b.jpg')
p=cv2.resize(p,(28,28))
p=np.reshape(p,(1,2352))
P=np.array(p)

data_path = 'train.tfrecords'







def init_weights(shape):
    init_random__dist=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(init_random__dist)

def init_bias(shape):
    init_bias__value=tf.constant(0.1,shape=shape)
    return tf.Variable(init_bias__value)

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2by2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')


def convolutional_layer(input_x,shape):
    W=init_weights(shape)
    b=init_bias([shape[3]])
    return tf.nn.relu(conv2d(input_x,W)+b)


def fully_connected_layer(input_layer,size):
    input_size=int(input_layer.get_shape()[1])
    W=init_weights([input_size,size])
    b=init_bias([size])
    return tf.matmul(input_layer,W)+b


x=tf.placeholder(tf.float32,shape=[None,2352])
y_true=tf.placeholder(tf.float32,shape=[None,2])
x_image=tf.reshape(x,[-1,28,28,3])

convo_1=convolutional_layer(x_image,shape=[5,5,3,32])

convo_1_pooling=max_pool_2by2(convo_1)

convo_2=convolutional_layer(convo_1_pooling,shape=[5,5,32,64])

convo_2_pooling=max_pool_2by2(convo_2)

convo_2_flat=tf.reshape(convo_2_pooling,[-1,7*7*64])

full_layer_one=tf.nn.relu(fully_connected_layer(convo_2_flat,1024))

hold_prop=tf.placeholder(tf.float32)

full_one_dropout=tf.nn.dropout(full_layer_one,keep_prob=hold_prop)

y_pred=fully_connected_layer(full_one_dropout,2)

cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_pred))

optimizer=tf.train.AdamOptimizer(learning_rate=0.001)

train=optimizer.minimize(cross_entropy)

init=tf.global_variables_initializer()


steps=5000
saver = tf.train.Saver()
with tf.Session() as sess :

   # sess.run(init)

    feature = {'train/image': tf.FixedLenFeature([], tf.string),
               'train/label': tf.FixedLenFeature([], tf.int64)}
    # Create a list of filenames and pass it to a queue
    filename_queue = tf.train.string_input_producer([data_path], num_epochs=1)
    # Define a reader and read the next record
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    # Decode the record read by the reader
    features = tf.parse_single_example(serialized_example, features=feature)
    # Convert the image data from string back to the numbers
    image = tf.decode_raw(features['train/image'], tf.float32)

    # Cast label data into int32
    label = tf.cast(features['train/label'], tf.int32)
    # Reshape image data into the original shape
    image = tf.reshape(image, [28,28,3])
     # tf.reshape(img, [tf.shape(x)[0], tf.shape(x)[1]*tf.shape(x)[2]*tf.shape(x)[3]])

    # Any preprocessing here ...

    # Creates batches by randomly shuffling tensors
    images, labels = tf.train.shuffle_batch([image, label], batch_size=10, capacity=30, num_threads=1,
                                            min_after_dequeue=10)


    # Initialize all global and local variables
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    # Create a coordinator and run all QueueRunner objects
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    saver.restore(sess, "./gesture_start")

    for batch_index in range(87):


        x_1 = []

        img, lbl = sess.run([images, labels])
        img = img.astype(np.uint8)
        for i in range(10):
            x_1.append(img[i].flatten())
        X=np.array(x_1)

        print(X.shape)
        lbl = np.array(lbl)
        lbl = np.reshape(lbl, (lbl.shape[0], 1))
        k=np.array([[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0]])
        #lbl=np.reshape(lbl,(10,1))


        #sess.run(train, feed_dict={x:P, y_true:k, hold_prop: 0.5})

        print("jj")

        if batch_index % 1 == 0:

            print("ON STEP : {}".format(batch_index))
           # print("LOSS IS ")
           # print(sess.run(y_pred, feed_dict={x:P, y_true:lbl, hold_prop: 1.0}))
            #print(sess.run(cross_entropy, feed_dict={x:P, y_true:k, hold_prop: 1.0}))
            print('\n')
            print("ACCUURACY : ")

            matches = tf.equal(tf.arg_max(y_pred, 1), tf.arg_max(y_true, 1))

            acc = tf.reduce_mean(tf.cast(matches, tf.float32))

            print(sess.run(acc, feed_dict={x:P, y_true:k, hold_prop: 1.0}))

            print('\n')


    coord.request_stop()

    # Wait for threads to stop
    coord.join(threads)
    #saver.save(sess, "./gesture_start")

    sess.close()



























