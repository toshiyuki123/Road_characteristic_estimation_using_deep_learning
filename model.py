import tensorflow.compat.v1 as tf
import tensorflow_addons as tfa
from functools import partial
import numpy as np
import argparse

tf.disable_v2_behavior()
tf.reset_default_graph()

parser = argparse.ArgumentParser()
parser.add_argument('--n_epoch', type=int, default=100)
parser.add_argument('--img_dir', type=str, default='./imgs')
parser.add_argument('--vib_dir', type=str, default='./vibs')
parser.add_argument('--save_dir', type=str, default='./save')

args = parser.parse_args()

X_train = np.load(args.img_dir+'/train.npy')
X_valid = np.load(args.img_dir+'/valid.npy')
y_train = np.load(args.vib_dir+'/train.npy')
y_valid = np.load(args.vib_dir+'/valid.npy')


def main():
    X = tf.placeholder(tf.float32, shape=(None, 150, 150, 3), name="X") 
    y = tf.placeholder(tf.float32, shape=(None, 3, 256, 1), name="y")
    training = tf.placeholder_with_default(False, shape=(), name='training')
    he_init = tf.variance_scaling_initializer()
    my_batch_norm_layer = partial(tf.layers.batch_normalization,
                              training=training,
                              momentum=0.9)
    my_conv2d_layer = partial(tf.layers.conv2d, kernel_initializer=he_init)
    my_dense_layer = partial(tf.layers.dense, kernel_initializer=he_init)
    my_dconv2d_layer = partial(tf.layers.conv2d_transpose, kernel_initializer=he_init)


    with tf.name_scope("encorder"):
        conv1 = my_conv2d_layer(X, filters=32, kernel_size=(20,20), strides=(2,2), padding='valid')
        bn1 = tf.nn.relu(my_batch_norm_layer(conv1))

        conv2 = my_conv2d_layer(bn1, filters=32, kernel_size=(20,20), strides=(2,2), padding='valid')
        bn2 = tf.nn.relu(my_batch_norm_layer(conv2))

        conv3 = my_conv2d_layer(bn2, filters=32, kernel_size=(4,4), strides=(2,2), padding='valid')
        bn3 = tf.nn.relu(my_batch_norm_layer(conv3))

        conv4 = my_conv2d_layer(bn3, filters=32, kernel_size=(4,4), strides=(2,2), padding='valid')
        bn4 = tf.nn.relu(my_batch_norm_layer(conv4))
        bn4_flat = tf.reshape(bn4, shape=[-1, 4*4*32])


    with tf.name_scope("dense"):
        fc1 = my_dense_layer(bn4_flat, 3)
        lbn1 = tf.nn.sigmoid(my_batch_norm_layer(fc1))

        fc2 = my_dense_layer(lbn1, 16)
        lbn2 = tf.nn.relu(my_batch_norm_layer(fc2))
        lbn2_flat = tf.reshape(lbn2, shape=[-1, 1, 16, 1])

    with tf.name_scope("decorder"):
        dconv1 = my_dconv2d_layer(lbn2_flat, filters=32, kernel_size=(1,5), strides=(3,2), padding='same')
        dbn1 = tf.nn.relu(my_batch_norm_layer(dconv1))

        dconv2 = my_dconv2d_layer(dbn1, filters=32, kernel_size=(2,5), strides=(1,2), padding='same')
        dbn2 = tf.nn.relu(my_batch_norm_layer(dconv2))

        dconv3 = my_dconv2d_layer(dbn2, filters=32, kernel_size=(2,6), strides=(1,2), padding='same')
        dbn3 = tf.nn.relu(my_batch_norm_layer(dconv3))

        dconv4 = my_dconv2d_layer(dbn3, filters=1, kernel_size=(2,6), strides=(1,2), padding='same')
        outputs = tf.nn.sigmoid(dconv4)

    with tf.name_scope("loss"):
        loss = tf.reduce_mean(tf.square(outputs - y))

    learning_rate = 0.01

    with tf.name_scope("train"):
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate,momentum=0.9, decay=0.9, epsilon=1e-10,centered=True)
        training_op = optimizer.minimize(loss)
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    n_epoch = args.n_epoch

    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    tensor_loss = {
        'train_loss': [],
        'valid_loss': []
        }

    print("epoch  train/loss  valid/loss")

    np.random.seed(seed=21)

    loss_global = 10

    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epoch):
            order0 = np.random.permutation(np.arange(0,70))
            order1 = np.random.permutation(np.arange(70,70*2))
            order2 = np.random.permutation(np.arange(70*2,70*3))
            order3 = np.random.permutation(np.arange(70*3,70*4))
            order4 = np.random.permutation(np.arange(70*4,70*5))
            order5 = np.random.permutation(np.arange(70*5,70*6))
            order6 = np.random.permutation(np.arange(70*6,70*7))
            order7 = np.random.permutation(np.arange(70*7,70*8))
            order8 = np.random.permutation(np.arange(70*8,70*9))

            loss_list = []
            index = np.zeros(9).astype('int64')

            for i in range(70):
                index[0] = order0[i]
                index[1] = order1[i]
                index[2] = order2[i]
                index[3] = order3[i]
                index[4] = order4[i]
                index[5] = order5[i]
                index[6] = order6[i]
                index[7] = order7[i]
                index[8] = order8[i]

                X_batch = X_train[index]
                y_batch = y_train[index]
                sess.run([training_op, extra_update_ops],
                         feed_dict={training: True, X: X_batch, y: y_batch})
                loss_batch = loss.eval(feed_dict={X: X_batch, y: y_batch})

                loss_list.append(loss_batch)

            loss_train = np.mean(loss_list)
            loss_valid = loss.eval(feed_dict={X: X_valid, y: y_valid})

            print('{0:>4d} {1:>10.4f} {2:>11.4f}'.format(epoch, loss_train, loss_valid))


            if loss_train <= loss_global:
                save_path = saver.save(sess, args.save_dir+'/model.ckpt')
                loss_global = loss_train


if __name__ == '__main__':
    main()
