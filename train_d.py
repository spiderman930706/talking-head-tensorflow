import time
import os
import io
import pickle as pkl
import multiprocessing
from datetime import datetime

import tensorflow as tf
import PIL
import numpy as np
import matplotlib.pyplot as plt

from module_keras import EmBedder, Generator, Discriminator, get_discriminator
from loss import loss_eg_fun, loss_d_fun

batch_size = 2


def plot_landmarks(frame, landmarks):
    """
    在生成landmarks的原图上绘制出landmarks的点
    """
    dpi = 100
    fig = plt.figure(figsize=(frame.shape[0] / dpi, frame.shape[1] / dpi), dpi=dpi)
    ax = fig.add_subplot(111)
    ax.axis('off')
    plt.imshow(np.ones(frame.shape))
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Head
    ax.plot(landmarks[0:17, 0], landmarks[0:17, 1], linestyle='-', color='green', lw=2)
    # Eyebrows
    ax.plot(landmarks[17:22, 0], landmarks[17:22, 1], linestyle='-', color='orange', lw=2)
    ax.plot(landmarks[22:27, 0], landmarks[22:27, 1], linestyle='-', color='orange', lw=2)
    # Nose
    ax.plot(landmarks[27:31, 0], landmarks[27:31, 1], linestyle='-', color='blue', lw=2)
    ax.plot(landmarks[31:36, 0], landmarks[31:36, 1], linestyle='-', color='blue', lw=2)
    # Eyes
    ax.plot(landmarks[36:42, 0], landmarks[36:42, 1], linestyle='-', color='red', lw=2)
    ax.plot(landmarks[42:48, 0], landmarks[42:48, 1], linestyle='-', color='red', lw=2)
    # Mouth
    ax.plot(landmarks[48:60, 0], landmarks[48:60, 1], linestyle='-', color='purple', lw=2)

    fig.canvas.draw()
    data = PIL.Image.frombuffer('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb(), 'raw', 'RGB', 0, 1)
    plt.close(fig)
    return data


def write_image(image, path):
    img = tf.image.encode_jpeg(tf.cast(image * 255, tf.uint8), format='rgb')
    tf.io.write_file(path, img)


def get_train_data():
    # load dataset
    train_hr_imgs_path = os.listdir('./parsed_video')

    def generator_train():
        for i, path in enumerate(train_hr_imgs_path):
            data = pkl.load(open(os.path.join('./parsed_video/', path), 'rb'))
            data_array = []
            for d in data:
                # x = PIL.Image.fromarray(d['frame'], 'RGB')
                # start_ = time.time()
                # y = plot_landmarks(d['frame'], d['landmarks'])
                x = d['frame']
                y = d['landmarks']
                # x = x.resize((256, 256))
                # y = y.resize((256, 256))
                # start_ = time.time()
                y = np.divide(y, 255, dtype=np.float32)
                x = np.divide(x, 255, dtype=np.float32)
                x = tf.constant(x, dtype=tf.float32)
                y = tf.constant(y, dtype=tf.float32)
                data_array.append(tf.stack([x, y]))  # 2, H, W, C
            video = tf.stack(data_array)  # 9, 2, H, W, C
            yield (i, video)

    shuffle_buffer_size = 200
    # shuffle_buffer_size = 2
    train_ds = tf.data.Dataset.from_generator(generator_train, output_types=(tf.int32, tf.float32))
    # train_ds = train_ds.map(_map_fn_train, num_parallel_calls=multiprocessing.cpu_count())
    train_ds = train_ds.shuffle(shuffle_buffer_size)
    train_ds = train_ds.prefetch(buffer_size=2)
    train_ds = train_ds.batch(batch_size, drop_remainder=True)
    # value = train_ds.make_one_shot_iterator().get_next()
    return train_ds


if not os.path.exists('./checkpoint'):
    os.mkdir('./checkpoint')
E = EmBedder()
G = Generator()
data_len = len(os.listdir('./parsed_video'))
# D = Discriminator(data_len)
D = get_discriminator((256, 256, 3))
LEARNING_RATE_E_G = 2e-5  #
LEARNING_RATE_D = 1e-4
optimizer_eg = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_E_G)
optimizer_d = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_D)


@tf.function
def train_step(num, video):
    with tf.GradientTape(persistent=True) as tape:
        t = video[:, -1, ...]  # [B, 2, C, H, W]
        video = video[:, :-1, ...]  # [B, K, 2, C, H, W]
        dims = video.shape
        # Calculate average encoding vector for video
        e_in = tf.reshape(video, shape=(
            dims[0] * dims[1], dims[2], dims[3], dims[4], dims[5]))  # [batch*帧数, 2, C, H, W]
        x, y = e_in[:, 0, ...], e_in[:, 1, ...]
        e_vectors = E(x, y)
        e_hat = tf.reshape(e_vectors, shape=(dims[0], dims[1], -1))  # B, K, len(e)
        e_hat = tf.keras.backend.mean(e_hat, axis=1)
        # generate
        x_t, y_t = t[:, 0, ...], t[:, 1, ...]
        x_hat = G(y_t, e_hat)
        # Optimize E_G and D
        r_x_hat = D(x_hat)
        r_x = D(x_t)
        # d_w = tf.gather(D.W, num, axis=1)  # 512, 2
        # d_w = tf.transpose(d_w, perm=[1, 0])
        adv, mean_square, cnt = loss_eg_fun(x_t, x_hat, r_x_hat, e_hat)
        loss_eg = adv + cnt + mean_square
        loss_d = loss_d_fun(r_x, r_x_hat)
    gradients = tape.gradient(loss_eg, G.trainable_variables)
    optimizer_eg.apply_gradients(zip(gradients, G.trainable_variables))
    gradients = tape.gradient(loss_eg, E.trainable_variables)
    optimizer_eg.apply_gradients(zip(gradients, E.trainable_variables))

    variables = D.trainable_variables
    gradients = tape.gradient(loss_d, variables)
    optimizer_d.apply_gradients(zip(gradients, variables))

    # with tf.GradientTape() as tape2:
    #     t = video[:, -1, ...]  # [B, 2, C, H, W]
    #     x_t, y_t = t[:, 0, ...], t[:, 1, ...]
    #     x_hat_ = G(y_t, e_hat)
    #     r_x_hat_ = D(x_hat_)
    #     r_x_ = D(x_t)
    #     loss_d2 = loss_d_fun(r_x_, r_x_hat_)
    # variables = D.trainable_variables
    # gradients = tape2.gradient(loss_d2, variables)
    # optimizer_d.apply_gradients(zip(gradients, variables))

    return loss_d, loss_eg, x_t, x_hat, r_x, r_x_hat, adv, mean_square, cnt, e_hat


@tf.function
def train_d_again(video, num):
    with tf.GradientTape(persistent=True) as tape:
        t = video[:, -1, ...]  # [B, 2, C, H, W]
        video = video[:, :-1, ...]  # [B, K, 2, C, H, W]
        dims = video.shape
        # Calculate average encoding vector for video
        e_in = tf.reshape(video, shape=(
            dims[0] * dims[1], dims[2], dims[3], dims[4], dims[5]))  # [batch*帧数, 2, C, H, W]
        x, y = e_in[:, 0, ...], e_in[:, 1, ...]
        e_vectors = E(x, y)
        e_hat = tf.reshape(e_vectors, shape=(dims[0], dims[1], -1))  # B, K, len(e)
        e_hat = tf.keras.backend.mean(e_hat, axis=1)
        # generate
        x_t, y_t = t[:, 0, ...], t[:, 1, ...]
        x_hat = G(y_t, e_hat)
        # Optimize E_G and D
        r_x_hat = D(x_hat)
        r_x = D(x_t)
        loss_d = loss_d_fun(r_x, r_x_hat)
    variables = D.trainable_variables
    gradients = tape.gradient(loss_d, variables)
    optimizer_d.apply_gradients(zip(gradients, variables))
    return loss_d, r_x_hat, r_x


def get_data():
    data = pkl.load(open('./extract_frame/e_vector.vid', 'rb'))
    data_array = []
    for d in data:
        x = d['frame']
        y = d['landmarks']
        y = np.divide(y, 255, dtype=np.float32)
        x = np.divide(x, 255, dtype=np.float32)
        x = tf.constant(x, dtype=tf.float32)
        y = tf.constant(y, dtype=tf.float32)
        data_array.append(tf.stack([x, y]))  # 2, H, W, C
    video = tf.stack(data_array)  # 9, 2, H, W, C
    video = video[tf.newaxis, ...]
    data = pkl.load(open('./extract_frame/land_mark.vid', 'rb'))
    y = data[0]['landmarks']
    y = np.divide(y, 255, dtype=np.float32)
    y = tf.constant(y, dtype=tf.float32)
    y = y[tf.newaxis, ...]
    return video, y


def generate(video, t, i):
    # t = video[:, -1, ...]  # [B, 2, C, H, W]      # t是真实的图片
    # video = video[:, :-1, ...]  # [B, K, 2, C, H, W]
    dims = video.shape
    # Calculate average encoding vector for video
    e_in = tf.reshape(video, shape=(
        dims[0] * dims[1], dims[2], dims[3], dims[4], dims[5]))  # [batch*帧数, 2, C, H, W]
    x, y = e_in[:, 0, ...], e_in[:, 1, ...]
    e_vectors = E(x, y)
    e_hat = tf.reshape(e_vectors, shape=(dims[0], dims[1], -1))  # B, K, len(e)
    e_hat = tf.keras.backend.mean(e_hat, axis=1)
    # generate
    y_t = t
    x_hat = G(y_t, e_hat)
    write_image(x_hat[0], './process_pic/x_hatresult{}.jpg'.format(i))


def train(start):
    epochs = 1000
    dataset = get_train_data()
    n_step_epoch = round(data_len // batch_size)
    pre_train_epoch = 10
    if not os.path.exists('./process_pic'):
        os.mkdir('./process_pic')
    e_path = './checkpoint/checkpointE/'
    g_path = './checkpoint/checkpointG/'
    d_path = './checkpoint/checkpointD/'
    if not os.path.exists(e_path):
        os.makedirs(e_path)
    if not os.path.exists(g_path):
        os.makedirs(g_path)
    if not os.path.exists(d_path):
        os.makedirs(d_path)
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = 'logs/' + current_time + '/loss'
    summary_writer = tf.summary.create_file_writer(log_dir)
    if start:
        E.load_weights('./checkpoint/checkpointE/')
        G.load_weights('./checkpoint/checkpointG/')
        D.load_weights('./checkpoint/checkpointD/')
        print('load weights end')
    evaluate_video, evaluate_mark = get_data()

    for epoch in range(start, epochs):
        epoch += 1
        epoch_time = time.time()
        total_d = 0
        total_eg = 0
        total_mean_square = 0
        total_adv = 0
        total_cnt = 0

        for step, (i, video) in enumerate(dataset):
            step += 1
            if video.shape[0] != batch_size:  # if the remaining data in this epoch < batch_size
                break
            step_time = time.time()
            loss_d, loss_eg, x_t, x_hat, r_x, r_x_hat, adv, mean_square, cnt, e_hat = train_step(i, video)
            # loss_d2 = train_d_again(video, i, e_hat)
            loss_d2 = 0
            print(
                "\rEpoch: [{}/{}] step: [{}/{}] time: {:.3f}s, loss_d: {:.3f}, loss_d2: {:.3f}, loss_eg: {:.3f}"
                "r_x:{:.4f}, r_x_hat:{:.4f}, adv:{:.4f}, mse:{:.4f}, cnt:{:.4f}".format(
                    epoch, epochs, step, n_step_epoch, time.time() - step_time, loss_d, loss_d2, loss_eg, r_x[0],
                    r_x_hat[0], adv, mean_square, cnt), end='')
            total_d += loss_d
            total_eg += loss_eg
            total_mean_square += mean_square
            total_adv += adv
            total_cnt += cnt
        with summary_writer.as_default():
            tf.summary.scalar('loss_d', total_d, step=epoch - 1)
            tf.summary.scalar('loss_eg', total_eg, step=epoch - 1)
        write_image(x_hat[0], './process_pic/x_hat{}.jpg'.format(epoch))
        generate(evaluate_video, evaluate_mark, epoch)
        # write_image(x_t[0], './process_pic/x_t{}.jpg'.format(epoch))
        if (epoch != 0) and (epoch % 10 == 0):
            E.save_weights(e_path)
            G.save_weights(g_path)
            D.save_weights(d_path)
            print('\rsave weights')
        print(
            '\rEpoch: [{}/{}],time: {:.4f}s, loss_d: {:.4f}, loss_eg: {:.4f}, total_mean_square:{:.4f}, total_adv:{:.4f}, total_cnt:{:.4f}'.format(
                epoch, epochs, time.time() - epoch_time, total_d, total_eg, total_mean_square, total_adv, total_cnt))


if __name__ == '__main__':
    train(100)

