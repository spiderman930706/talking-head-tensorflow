import io

import tensorflow as tf

match_loss = False

LOSS_VGG_FACE_WEIGHT = 2e-3
LOSS_VGG19_WEIGHT = 1e-2
LOSS_MCH_WEIGHT = 8e1
LOSS_FM_WEIGHT = 1e1

base_model = tf.keras.applications.VGG19(include_top=False)
layer_name = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']


def vgg_layer_result(x):
    result = []
    for layer in base_model.layers:
        x = layer(x)
        if layer.name in layer_name:
            result.append(x)
    return result


def loss_adv(r_x_hat):
    # return -tf.reduce_mean(r_x_hat)
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(r_x_hat), logits=r_x_hat))


def loss_mch(e_hat, W_i):
    abs_value = tf.abs(tf.reshape(W_i, shape=(-1,)) - tf.reshape(e_hat, shape=(-1,)))
    return tf.reduce_mean(abs_value) * LOSS_MCH_WEIGHT


def loss_vgg(x, x_hat):
    result = vgg_layer_result(x)
    result_hat = vgg_layer_result(x_hat)
    vgg19_loss = 0
    for i in range(len(result)):
        abs_value = tf.abs(result[i] - result_hat[i])
        vgg19_loss += tf.reduce_mean(abs_value)
    return vgg19_loss * LOSS_VGG19_WEIGHT


def loss_eg_fun(x, x_hat, r_x_hat, e_hat):
    cnt = loss_vgg(x, x_hat)
    adv = loss_adv(r_x_hat)
    mean_square = tf.reduce_mean(tf.square(x - x_hat))
    return adv * 0.1, mean_square * 100, cnt * 10


def loss_d_fun(r_x, r_x_hat):
    # return tf.reduce_mean(tf.nn.relu(1 + r_x_hat) + tf.nn.relu(1 - r_x))
    real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(r_x), logits=r_x))
    fake_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(r_x_hat), logits=r_x_hat))
    return real_loss + fake_loss
