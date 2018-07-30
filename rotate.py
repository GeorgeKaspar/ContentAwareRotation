import tensorflow as tf
import math
import numpy as np
from grid import fit_lines_to_grid, draw_grid, test_fit_line_to_grid, draw_line, get_mesh_grid
import cv2
import matplotlib.pyplot as plt
from utils import get_image_ratio, get_theta_by_lines
from lsd import lsd
from utils import get_bin_by_theta, get_quad_index_basis_by_lines, get_theta_index_by_lines, get_theta_by_lines, get_rotation_matrices
from quad import get_Aq, get_Vq, grid_to_quads
#angle belongs to (-90 90)
# HELPERS

def l2(x):
    return tf.reduce_sum(x * x, axis=[-1, -2])

def fn(x, R=None):
    y = tf.matrix_inverse(tf.matmul(x, x, transpose_a=True))
    x = tf.matmul(tf.matmul(x, y), x, transpose_b=True)
    if R is not None:
        x = tf.matmul(tf.matmul(R, x), R, transpose_b=True) 
    shape = x.get_shape().as_list()
    print(shape)
    return x - tf.eye(shape[1], batch_shape=[tf.shape(x)[0]])


# ENERGY
def rotation_energy(theta, angle):
    near_ninety = 90
    m = get_bin_by_theta(theta, angle)
    theta = tf.Print(theta, ["theta", theta[0]])
    return (10**3) * (((theta[0]) ** 2) + ((theta[89] - math.pi) ** 2) + ((theta[44] - math.pi / 2) ** 2) + ((theta[45] - math.pi / 2) ** 2))  + tf.reduce_sum((tf.manip.roll(theta, 1, 0) - theta) ** 2)

def boundary_preservation_energy(R, C):
    return tf.reduce_sum(R[:, 0] ** 2) + tf.reduce_sum((R[:, -1] - 1)**2) + tf.reduce_sum(C[0, :] ** 2) + tf.reduce_sum((C[-1, :] - 1)**2)

def line_preservation_energy(theta, N, R, C, lines, angle):

    u = tf.expand_dims(lines[:, 2:] - lines[:, :2], axis=-1)

    theta_index = get_theta_index_by_lines(lines, angle)
    lines_theta = tf.gather(theta, theta_index)
    M = get_rotation_matrices(lines_theta)

    quads_index, coeff, v, v1, v2 = get_quad_index_basis_by_lines(lines, N)
    quads = grid_to_quads(R, C)
    e = tf.expand_dims(tf.reduce_sum(coeff * tf.gather(quads, quads_index), axis=[1,2]), axis=-1)

    return tf.reduce_mean(l2(tf.matmul(fn(u, R=M), e)))

    
def shape_preservation_energy(theta, R, C, R0, C0):
    Aq = get_Aq(R, C)
    Vq = get_Vq(R0, C0)
    return tf.reduce_mean(l2(tf.matmul(fn(Aq), Vq)))

def show_image_v(image, R, C):
    B = np.reshape(np.stack([R, C], axis=-1), (-1, 2))
    for i in range(B.shape[0]):
        cv2.circle(image, (int(image.shape[1] * B[i, 1]), int(image.shape[0] * B[i, 0])), 3, (255, 0, 0))
    plt.imshow(image)
    plt.show()


def warp(image, R0, C0, R, C):
    shape = image.get_shape().as_list()
    A = tf.cast(tf.reshape(tf.stack([shape[0] * R0, shape[1] * C0], axis=-1), (1, -1, 2)), tf.float32)
    B = tf.cast(tf.reshape(tf.stack([shape[0] * R, shape[1] * C], axis=-1), (1, -1, 2)), tf.float32)
    image = tf.expand_dims(tf.cast(image, tf.float32) / 255.0, axis=0)
    image, _ = tf.contrib.image.sparse_image_warp(image, A, B)
    image = tf.cast(image * 255, tf.uint8)
    return image[0]

def model_fn(image, angle, N, lambda_s=1e-7, lambda_b=10, lambda_l=1e-5, lambda_r=1e-5):
    import matplotlib.pyplot as plt
    lines = lsd(image, N)

    R0, C0 = get_mesh_grid(N)

    theta_lines = get_theta_by_lines(lines, angle)
    theta_index = get_bin_by_theta(theta_lines, angle)
    theta0 = tf.scatter_add(tf.Variable(np.zeros(90), trainable=False, dtype=tf.float32), theta_index, theta_lines) / tf.scatter_add(tf.Variable(np.zeros(90), trainable=False, dtype=tf.float32), theta_index, tf.ones_like(theta_lines))

    theta = tf.get_variable('theta', shape=[90], dtype=tf.float32, trainable=True)
    R = tf.get_variable('R', shape=R0.get_shape().as_list(), dtype=tf.float32, trainable=True)
    C = tf.get_variable('C', shape=C0.get_shape().as_list(), dtype=tf.float32, trainable=True)
    assign_op = tf.group(tf.assign(theta, theta0), tf.assign(R, R0), tf.assign(C, C0))
    loss =  lambda_s * shape_preservation_energy(theta, R, C, R0, C0) + \
            lambda_b * boundary_preservation_energy(R, C) + \
            lambda_l * line_preservation_energy(theta, N, R, C, lines, angle) + \
            lambda_r * rotation_energy(theta, angle)
    shape = image.get_shape().as_list()
    flow = tf.stack([(R - R0), (C - C0)], axis=-1)
    movement = tf.reduce_mean(tf.sqrt(tf.reduce_sum(flow ** 2, axis=-1)))
    new_image = warp(image, R0, C0, R, C) #tf.cast(tf.contrib.image.dense_image_warp(tf.cast(tf.expand_dims(image, axis=0), tf.float32), flow)[0], tf.uint8) # TODO sparse_image_warp
    opt = tf.train.AdamOptimizer(1e-1)
    train_op = opt.minimize(loss, var_list=[theta, R, C])
    with tf.Session() as sess:
        _image_in = sess.run(image)
        plt.imshow(_image_in)
        plt.show()
        sess.run(tf.global_variables_initializer())
        sess.run(assign_op)
        print("opt by theta")
        for i in range(2000):
            _, _loss, _movement = sess.run([train_op, loss, movement])
            print("loss: %f, movement: %f" % (_loss, _movement))
            if i % 100 == 0:
                _image = sess.run(new_image)
                plt.imshow(np.concatenate([_image, _image_in], axis=1))
                plt.show()

        print(sess.run(theta))

def main(path):
    image = tf.constant(cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB), dtype=tf.uint8)
    angle = math.pi / 6
    N = [20, 20]
    model_fn(image, angle, N)


# TESTS

def test_rotation_energy():
    theta = tf.placeholder(tf.float32, shape=[90])
    angle = tf.placeholder(tf.float32, shape=[])
    energy = rotation_energy(theta, angle)
    sess = tf.InteractiveSession()
    print(sess.run(energy, feed_dict={theta : np.zeros(90), angle : math.pi / 2}))


def test_warp():
    N = [20, 20]
    path = '/media/george/1a4f4334-123f-430d-8a2b-f0c0fa401c75/places2_hr/data_large/b/bar/00000001.jpg' 
    image = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    R0, C0 = np.meshgrid(np.linspace(0, 1, N[0] + 1), np.linspace(0, 1, N[1] + 1))
    R, C = R0 + np.random.randn(*R0.shape) / 100, C0 + np.random.randn(*C0.shape) / 100
    show_image_v(image, R0, C0)
    show_image_v(image, R, C)
    sess = tf.InteractiveSession()
    print(image)
    new_image = sess.run(warp(image, R0, C0, R, C))
    print(new_image)
    show_image_v(np.uint8(new_image), R, C)

     
if __name__ == '__main__':
    #test_rotation_energy()
    #test_warp()
    #'''
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    args = parser.parse_args()
    main(args.path)
    #'''
