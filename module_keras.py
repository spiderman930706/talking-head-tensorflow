import io
import pickle as pkl
from collections import OrderedDict

import PIL
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

import loss

E_VECTOR_LENGTH = 512


class AdaptiveResidualBlockUp(tf.keras.models.Model):
    def __init__(self, channels, kernel_size=3, stride=1, upsample=2):
        super(AdaptiveResidualBlockUp, self).__init__()

        # General
        # self.upsample = nn.Upsample(scale_factor=upsample, mode='nearest')
        self.upsample = tf.keras.layers.UpSampling2D(size=upsample)

        # Right Side
        self.norm_r1 = AdaIn()
        self.conv_r1 = tf.keras.layers.Conv2D(channels, kernel_size, strides=stride, padding='same')
        self.norm_r2 = AdaIn()
        self.conv_r2 = tf.keras.layers.Conv2D(channels, kernel_size, strides=stride, padding='same')
        self.batch1 = tf.keras.layers.BatchNormalization()  # 这里用的批归一化， 跟原版用的归一化不一样
        self.batch2 = tf.keras.layers.BatchNormalization()

        # Left Side
        self.conv_l = tf.keras.layers.Conv2D(channels, 1, strides=1, padding='same')
        self.batch3 = tf.keras.layers.BatchNormalization()

    def call(self, x, mean1, std1, mean2, std2):
        residual = x

        # Right Side
        out = self.norm_r1(x, mean1, std1)
        out = tf.nn.relu(out)
        out = self.upsample(out)
        out = self.conv_r1(out)
        out = self.batch1(out)
        out = self.norm_r2(out, mean2, std2)
        out = tf.nn.relu(out)
        out = self.conv_r2(out)
        out = self.batch2(out)

        # Left Side
        residual = self.upsample(residual)
        residual = self.conv_l(residual)
        residual = self.batch3(residual)

        # Merge
        out = residual + out
        return out


class AdaptiveResidualBlock(tf.keras.models.Model):
    def __init__(self, channels):
        super(AdaptiveResidualBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(channels, 3, strides=1, padding='same')
        self.conv2 = tf.keras.layers.Conv2D(channels, 3, strides=1, padding='same')

        self.batch1 = tf.keras.layers.BatchNormalization()  # 这里用的批归一化， 跟原版用的归一化不一样
        self.batch2 = tf.keras.layers.BatchNormalization()

        self.in1 = AdaIn()
        self.in2 = AdaIn()

    def call(self, x, mean1, std1, mean2, std2):
        residual = x

        out = self.conv1(x)  # [B, 4, 4, 512]
        out = self.in1(out, mean1, std1)
        out = self.batch1(out)
        out = tf.nn.relu(out)
        out = self.conv2(out)
        out = self.batch2(out)
        out = self.in2(out, mean1, std1)

        out = out + residual
        return out


class AdaIn(tf.keras.models.Model):
    def __init__(self):
        super(AdaIn, self).__init__()
        self.eps = 1e-5

    def call(self, x, mean_style, std_style):
        B, H, W, C = x.shape

        feature = tf.reshape(x, shape=(B, H * W, C))
        # std_feat = (np.std(feature.numpy(), axis=1) + self.eps)
        std_feat = (tf.keras.backend.std(feature, axis=1) + self.eps)
        std_feat = tf.reshape(std_feat, shape=(B, 1, C))
        # mean_feat = np.mean(feature.numpy(), axis=1)
        mean_feat = tf.keras.backend.mean(feature, axis=1)
        mean_feat = tf.reshape(mean_feat, shape=(B, 1, C))

        adain = std_style * (feature - mean_feat) / std_feat + mean_style

        adain = tf.reshape(adain, shape=(B, H, W, C))
        return adain


class MyDense(tf.keras.layers.Layer):
    ADAIN_LAYERS = OrderedDict([
        ('res1', (512, 512)),
        ('res2', (512, 512)),
        ('res3', (512, 512)),
        ('res4', (512, 512)),
        ('res5', (512, 512)),
        ('deconv6', (512, 512)),
        ('deconv5', (512, 512)),
        ('deconv4', (512, 256)),
        ('deconv3', (256, 128)),
        ('deconv2', (128, 64)),
        ('deconv1', (64, 3))
    ])

    def __init__(self):
        super(MyDense, self).__init__()
        self.PSI_PORTIONS, self.psi_length = self.define_psi_slices()

    def build(self, input_shape):
        self.projection = self.add_weight(
            name='projection', shape=(self.psi_length, E_VECTOR_LENGTH),
            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02), trainable=True)

    def call(self, e):
        P = self.projection[tf.newaxis, ...]  # 1, 17158, 512
        P = tf.tile(P, [e.shape[0], 1, 1])  # B, 17158, 512
        mat = tf.matmul(P, e[..., tf.newaxis])  # 1, 17158, 1
        psi_hat = tf.squeeze(mat, [2])
        return psi_hat

    def define_psi_slices(self):
        out = {}
        d = self.ADAIN_LAYERS
        start_idx, end_idx = 0, 0
        for layer in d:
            end_idx = start_idx + d[layer][0] * 2 + d[layer][1] * 2
            out[layer] = (start_idx, end_idx)
            start_idx = end_idx

        return out, end_idx


class Generator(tf.keras.models.Model):
    ADAIN_LAYERS = OrderedDict([
        ('res1', (512, 512)),
        ('res2', (512, 512)),
        ('res3', (512, 512)),
        ('res4', (512, 512)),
        ('res5', (512, 512)),
        ('deconv6', (512, 512)),
        ('deconv5', (512, 512)),
        ('deconv4', (512, 256)),
        ('deconv3', (256, 128)),
        ('deconv2', (128, 64)),
        ('deconv1', (64, 3))
    ])

    def __init__(self):
        super(Generator, self).__init__()

        self.PSI_PORTIONS, self.psi_length = self.define_psi_slices()
        self.conv1 = ResidualBlockDown(64)
        self.in1_e = tfa.layers.InstanceNormalization()

        self.conv2 = ResidualBlockDown(128)
        self.in2_e = tfa.layers.InstanceNormalization()

        self.conv3 = ResidualBlockDown(256)
        self.in3_e = tfa.layers.InstanceNormalization()

        self.att1 = SelfAttention(256)

        self.conv4 = ResidualBlockDown(512)
        self.in4_e = tfa.layers.InstanceNormalization()

        self.conv5 = ResidualBlockDown(512)
        self.in5_e = tfa.layers.InstanceNormalization()

        self.conv6 = ResidualBlockDown(512)
        self.in6_e = tfa.layers.InstanceNormalization()

        self.res1 = AdaptiveResidualBlock(512)
        self.res2 = AdaptiveResidualBlock(512)
        self.res3 = AdaptiveResidualBlock(512)
        self.res4 = AdaptiveResidualBlock(512)
        self.res5 = AdaptiveResidualBlock(512)

        self.deconv6 = AdaptiveResidualBlockUp(512, upsample=2)
        self.in6_d = tfa.layers.InstanceNormalization()

        self.deconv5 = AdaptiveResidualBlockUp(512, upsample=2)
        self.in5_d = tfa.layers.InstanceNormalization()

        self.deconv4 = AdaptiveResidualBlockUp(256, upsample=2)
        self.in4_d = tfa.layers.InstanceNormalization()

        self.deconv3 = AdaptiveResidualBlockUp(128, upsample=2)
        self.in3_d = tfa.layers.InstanceNormalization()

        self.att2 = SelfAttention(128)

        self.deconv2 = AdaptiveResidualBlockUp(64, upsample=2)
        self.in2_d = tfa.layers.InstanceNormalization()

        self.deconv1 = AdaptiveResidualBlockUp(3, upsample=2)
        self.in1_d = tfa.layers.InstanceNormalization()

        self.projection_layer = MyDense()

    def call(self, y, e):
        out = y  # [B, 256, 256, 3]
        psi_hat = self.projection_layer(e)
        # Encode
        out = self.conv1(out)  # 1, 112, 112, 64
        out = self.in1_e(out)  # [B, 64, 128, 128]
        out = self.conv2(out)
        out = self.in2_e(out)  # [B, 128, 64, 64]
        out = self.conv3(out)
        out = self.in3_e(out)  # [B, 256, 32, 32]
        out = self.att1(out)
        out = self.conv4(out)
        out = self.in4_e(out)  # [B, 512, 16, 16]
        out = self.conv5(out)
        out = self.in5_e(out)  # [B, 512, 8, 8]
        out = self.conv6(out)
        out = self.in6_e(out)  # [B, 512, 4, 4]

        # Residual layers
        out = self.res1(out, *self.slice_psi(psi_hat, 'res1'))
        out = self.res2(out, *self.slice_psi(psi_hat, 'res2'))
        out = self.res3(out, *self.slice_psi(psi_hat, 'res3'))
        out = self.res4(out, *self.slice_psi(psi_hat, 'res4'))
        out = self.res5(out, *self.slice_psi(psi_hat, 'res5'))

        # Decode
        out = self.deconv6(out, *self.slice_psi(psi_hat, 'deconv6'))
        out = self.in6_d(out)  # [B, 512, 4, 4]
        out = self.deconv5(out, *self.slice_psi(psi_hat, 'deconv5'))
        out = self.in5_d(out)  # [B, 512, 16, 16]
        out = self.deconv4(out, *self.slice_psi(psi_hat, 'deconv4'))
        out = self.in4_d(out)  # [B, 256, 32, 32]
        out = self.deconv3(out, *self.slice_psi(psi_hat, 'deconv3'))
        out = self.in3_d(out)  # [B, 128, 64, 64]
        out = self.att2(out)
        out = self.deconv2(out, *self.slice_psi(psi_hat, 'deconv2'))
        out = self.in2_d(out)  # [B, 64, 128, 128]
        out = self.deconv1(out, *self.slice_psi(psi_hat, 'deconv1'))
        out = self.in1_d(out)  # [B, 3, 256, 256]

        out = tf.nn.tanh(out)

        return out

    def slice_psi(self, psi, portion):
        idx0, idx1 = self.PSI_PORTIONS[portion]
        len1, len2 = self.ADAIN_LAYERS[portion]
        aux = psi[:, tf.newaxis, idx0:idx1]
        mean1, std1 = aux[:, :, 0:len1], aux[:, :, len1:2 * len1]
        mean2, std2 = aux[:, :, 2 * len1:2 * len1 + len2], aux[:, :, 2 * len1 + len2:]
        return mean1, std1, mean2, std2

    def define_psi_slices(self):
        out = {}
        d = self.ADAIN_LAYERS
        start_idx, end_idx = 0, 0
        for layer in d:
            end_idx = start_idx + d[layer][0] * 2 + d[layer][1] * 2
            out[layer] = (start_idx, end_idx)
            start_idx = end_idx

        return out, end_idx


class EmBedder(tf.keras.models.Model):
    def __init__(self):
        super(EmBedder, self).__init__()
        self.conv1 = ResidualBlockDown(64)
        self.conv2 = ResidualBlockDown(128)
        self.conv3 = ResidualBlockDown(256)
        self.att = SelfAttention(256)
        self.conv4 = ResidualBlockDown(512)
        self.conv5 = ResidualBlockDown(512)
        self.conv6 = ResidualBlockDown(512)
        # self.pooling = nn.AdaptiveMaxPool2d((1, 1))

    def call(self, x, y):
        out = tf.concat([x, y], axis=-1)

        # Encode
        out = self.conv1(out)  # [BxK, 128, 128, 64]
        out = self.conv2(out)  # [BxK, 64, 64, 128]
        out = self.conv3(out)  # [BxK, 32, 32, 256]
        out = self.att(out)
        out = self.conv4(out)  # [BxK, 16, 16, 512]
        out = self.conv5(out)  # [BxK, 8, 8, 512]
        out = self.conv6(out)  # [BxK, 4, 4, 512]

        # Vectorize
        B, H, W, C = out.shape
        out = tf.nn.max_pool(out, [1, H, W, 1], [1, 1, 1, 1], padding='VALID')
        out = tf.reshape(out, shape=(-1, E_VECTOR_LENGTH))
        out = tf.nn.relu(out)
        return out


class ResidualBlockDown(tf.keras.models.Model):
    def __init__(self, filters, kernel_size=3, stride=1, padding='same'):
        super(ResidualBlockDown, self).__init__()
        # 右边
        self.con1 = tf.keras.layers.Conv2D(filters, kernel_size, strides=stride, padding=padding)
        self.con2 = tf.keras.layers.Conv2D(filters, kernel_size, strides=stride, padding=padding)
        self.batch1 = tf.keras.layers.BatchNormalization()
        self.batch2 = tf.keras.layers.BatchNormalization()

        # 左边
        self.con_l = tf.keras.layers.Conv2D(filters, kernel_size, strides=stride, padding=padding)

    def call(self, x):
        residual = x
        # Right Side
        out = self.con1(x)
        out = self.batch1(out)
        out = tf.nn.relu(out)
        out = self.con2(out)
        out = self.batch2(out)
        out = tf.nn.relu(out)
        out = tf.keras.layers.AveragePooling2D(2)(out)

        # Left Side
        residual = self.con_l(residual)
        residual = tf.keras.layers.AveragePooling2D(2)(residual)

        # Merge
        out = tf.add(residual, out)
        return out


class Gamma(tf.keras.layers.Layer):
    def __init__(self):
        super(Gamma, self).__init__()

    def build(self, input_shape):
        self.gamma = self.add_weight(
            name='gamma', shape=(), initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02), trainable=True)

    def call(self, out, x):
        out = self.gamma * out + x
        return out


class SelfAttention(tf.keras.models.Model):
    def __init__(self, filters, kernel_size=3, stride=1, padding='same'):
        super(SelfAttention, self).__init__()
        self.query_conv = tf.keras.layers.Conv2D(filters // 8, kernel_size, strides=stride, padding=padding)
        self.key_conv = tf.keras.layers.Conv2D(filters // 8, kernel_size, strides=stride, padding=padding)
        self.value_conv = tf.keras.layers.Conv2D(filters, kernel_size, strides=stride, padding=padding)
        self.gamma = Gamma()
        # self.softmax = nn.Softmax(dim=-1)

    def call(self, x):
        B, H, W, C = x.shape  # TensorFlow中的shape不一样
        proj_query = tf.reshape(self.query_conv(x), shape=(B, W * H, -1))  # 8, 28*28,
        proj_query = tf.transpose(proj_query, perm=[0, 2, 1])  # 8,32,28*28
        proj_key = tf.reshape(self.key_conv(x), shape=(B, W * H, -1))  # 8,28*28,32
        energy = tf.matmul(proj_key, proj_query)  # 8, 28*28, 28*28
        attention = tf.nn.softmax(energy)
        attention = tf.transpose(attention, perm=[0, 2, 1])  # 8, 28*28, 28*28
        proj_value = tf.reshape(self.value_conv(x), shape=(B, W * H, -1))  # 8, 28*28, 256
        out = tf.matmul(attention, proj_value)
        out = tf.reshape(out, shape=(B, H, W, -1))
        out = self.gamma(out, x)
        return out


class ResidualBlock(tf.keras.models.Model):
    def __init__(self, filters, kernel_size=3, stride=1, padding='same'):
        super(ResidualBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters, kernel_size, strides=stride, padding=padding)
        self.in1 = tfa.layers.InstanceNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters, kernel_size, strides=stride, padding=padding)
        self.in2 = tfa.layers.InstanceNormalization()

    def call(self, x):
        residual = x

        out = self.conv1(x)
        out = self.in1(out)
        out = tf.nn.relu(out)
        out = self.conv2(out)
        out = self.in2(out)
        out = tf.nn.relu(out)

        out = out + residual
        return out


def get_discriminator(input_shape):
    input_layer = tf.keras.Input(shape=input_shape, batch_size=2)
    n = tf.keras.layers.Conv2D(64, 3, strides=2, padding='same')(input_layer)       # 128
    n = tfa.layers.InstanceNormalization()(n)
    n = tf.nn.relu(n)
    n = tf.keras.layers.Conv2D(64, 3, strides=1, padding='same')(n)         # 128
    n = tfa.layers.InstanceNormalization()(n)
    n = tf.nn.relu(n)
    n = tf.keras.layers.Conv2D(128, 3, strides=2, padding='same')(n)        # 64
    n = tfa.layers.InstanceNormalization()(n)
    n = tf.nn.relu(n)
    n = tf.keras.layers.Conv2D(128, 3, strides=1, padding='same')(n)        # 64
    n = tfa.layers.InstanceNormalization()(n)
    n = tf.nn.relu(n)
    n = tf.keras.layers.Conv2D(256, 3, strides=2, padding='same')(n)  # 32
    n = tfa.layers.InstanceNormalization()(n)
    n = tf.nn.relu(n)
    # n = tf.keras.layers.Conv2D(256, 3, strides=1, padding='same')(n)  # 32
    # n = tfa.layers.InstanceNormalization()(n)
    # n = tf.nn.relu(n)
    # n = tf.keras.layers.Conv2D(512, 3, strides=2, padding='same')(n)  # 16
    # n = tfa.layers.InstanceNormalization()(n)
    # n = tf.nn.relu(n)
    # n = tf.keras.layers.Conv2D(512, 3, strides=1, padding='same')(n)  # 16
    # n = tfa.layers.InstanceNormalization()(n)
    # n = tf.nn.relu(n)
    n = tf.keras.layers.Conv2D(256, 3, strides=1, padding='same')(n)  # 16
    n = tfa.layers.InstanceNormalization()(n)
    n = tf.nn.relu(n)
    n = tf.keras.layers.Conv2D(128, 3, strides=1, padding='same')(n)  # 16
    n = tfa.layers.InstanceNormalization()(n)
    n = tf.nn.relu(n)
    n = tf.keras.layers.Conv2D(64, 3, strides=1, padding='same')(n)  # 16
    n = tfa.layers.InstanceNormalization()(n)
    n = tf.nn.relu(n)
    n = tf.keras.layers.Conv2D(32, 3, strides=1, padding='same')(n)  # 16
    n = tfa.layers.InstanceNormalization()(n)
    n = tf.nn.relu(n)
    n = tf.keras.layers.Flatten()(n)
    out = tf.keras.layers.Dense(1, activation='sigmoid')(n)
    out = tf.reshape(out, shape=(input_layer.shape[0], ))
    model = tf.keras.Model(inputs=input_layer, outputs=out)
    return model


class Discriminator(tf.keras.models.Model):
    def __init__(self, training_videos):
        super(Discriminator, self).__init__()

        self.conv1 = ResidualBlockDown(64)
        self.conv2 = ResidualBlockDown(128)
        self.conv3 = ResidualBlockDown(256)
        self.att = SelfAttention(256)
        self.conv4 = ResidualBlockDown(512)
        self.conv5 = ResidualBlockDown(512)
        self.conv6 = ResidualBlockDown(512)
        self.res_block = ResidualBlock(512)
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, x, y, i):
        assert x.shape == y.shape, "Both x and y must be tensors with shape [BxK, W, H, 3]."

        out = x

        # Encode
        out_0 = self.conv1(out)  # [B, 64, 128, 128]
        out_1 = self.conv2(out_0)  # [B, 128, 64, 64]
        out_2 = self.conv3(out_1)  # [B, 256, 32, 32]
        out_3 = self.att(out_2)
        out_4 = self.conv4(out_3)  # [B, 512, 16, 16]
        out_5 = self.conv5(out_4)  # [B, 512, 8, 8]
        out_6 = self.conv6(out_5)  # [B, 512, 4, 4]
        out_7 = self.res_block(out_6)

        x = self.flatten(out_7)
        out = self.dense(x)

        out = tf.reshape(out, shape=(x.shape[0],))
        return out, [out_0, out_1, out_2, out_3, out_4, out_5, out_6, out_7]


def plot_landmarks(frame, landmarks):
    """
    在生成landmarks的原图上绘制出landmarks的点

    :param frame: Image with a face matching the landmarks.
    :param landmarks: Landmarks of the provided frame,
    :return: RGB image with the landmarks as a Pillow Image.
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


def write_image(image):
    img = tf.image.encode_jpeg(tf.cast(image * 255, tf.uint8), format='rgb')
    tf.io.write_file('./test.jpg', img)


if __name__ == '__main__':
    # embed
    # tensorflow 的图片格式 B, H, W, C
    # video [B, K+1, 2, C, H, W]
    data = pkl.load(open('./parsed_video/2DLq_Kkc1r8.vid', 'rb'))

    data_array = []
    for d in data:
        # x = PIL.Image.fromarray(d['frame'], 'RGB')
        # y = plot_landmarks(d['frame'], d['landmarks'])
        x = d['frame']
        y = d['landmarks']
        # x = x.resize((256, 256))
        # y = y.resize((256, 256))
        # x = np.array(x).astype(np.uint8)    # tensorflow不知道怎么直接转换
        # y = np.array(y).astype(np.uint8)
        y = np.divide(y, 255, dtype=np.float32)
        x = np.divide(x, 255, dtype=np.float32)
        x = tf.constant(x, dtype=tf.float32)
        y = tf.constant(y, dtype=tf.float32)
        data_array.append(tf.stack([x, y]))  # 2, H, W, C
    video = tf.stack(data_array)  # 9, 2, H, W, C
    # video = video[tf.newaxis, :]
    video = tf.stack([video, video])
    # Put one frame aside (frame t)
    t = video[:, -1, ...]  # [B, 2, C, H, W]        #
    video = video[:, :-1, ...]  # [B, K, 2, C, H, W]
    dims = video.shape

    # Calculate average encoding vector for video
    e_in = tf.reshape(video, shape=(dims[0] * dims[1], dims[2], dims[3], dims[4], dims[5]))  # [batch*帧数, 2, C, H, W]
    x, y = e_in[:, 0, ...], e_in[:, 1, ...]
    E = EmBedder()
    e_vectors = E(x, y)
    e_hat = tf.reshape(e_vectors, shape=(dims[0], dims[1], -1))  # B, K, len(e)
    print(e_hat.shape)
    e_hat = np.mean(e_hat.numpy(), axis=1)
    # generate
    G = Generator()
    x_t, y_t = t[:, 0, ...], t[:, 1, ...]
    x_hat = G(y_t, e_hat)
    # loss.loss_vgg(x_t, x_hat)
    # print(x_hat)
    # D = Discriminator(100)
    # # Optimize E_G and D
    # r_x_hat, _ = D(x_hat, y_t, [0, 1])
    # r_x, _ = D(x_t, y_t, [0, 1])

    D = get_discriminator((256,256,3))
    r_x_hat = D(x_hat)
    r_x = D(x_t)
    print(r_x)
    print(r_x_hat)
