import tensorflow as tf
import numpy as np
import math


def get_image_ratio(image):
    shape = tf.cast(tf.shape(image), tf.float32)
    ratio = shape[1] / shape[0]
    return ratio

def get_theta_by_lines(lines, angle):
    v = lines[:, 2:] - lines[:, :2]
    theta = tf.atan(v[:, 0] / v[:, 1])
    return theta + math.pi * tf.floor(1 - (theta + angle) / math.pi)


def get_theta_index_by_lines(lines, angle):
    lines_theta = get_theta_by_lines(lines, angle)
    return get_bin_by_theta(lines_theta, angle)

def get_rotation_matrices(theta):
    cos = tf.cos(theta)
    sin = tf.sin(theta)
    return tf.reshape(tf.stack([cos, -sin, sin, cos], axis=-1), (-1, 2, 2))

def get_bin_by_theta(theta, angle):
    return tf.cast(90 * (theta + angle) / math.pi, tf.int64)


def get_quad_index_basis_by_lines(lines, N):
    r = tf.reduce_min(lines[:, ::2] * N[0], axis=1, keepdims=False)
    c = tf.reduce_min(lines[:, 1::2] * N[1], axis=1, keepdims=False)
    ir = tf.cast(r, tf.int64)
    ic = tf.cast(c, tf.int64)
    irf = tf.cast(ir, tf.float32)
    icf = tf.cast(ic, tf.float32)

    index = ir * N[1] + ic
    v = tf.stack([irf / N[0], icf / N[1]], axis=-1)
    v1 = tf.stack([(irf + 1) / N[0], icf / N[1]], axis=-1)
    v2 = tf.stack([irf / N[0], (icf + 1) / N[1]], axis=-1)
    
    alpha1, alpha2 = tf.split(lines[:, :2] - v, 2, axis=-1)
    beta1, beta2 = tf.split(lines[:, 2:] - v, 2, axis=-1)
    alpha1 = N[0] * alpha1
    alpha2 = N[1] * alpha2
    beta1 = N[0] * beta1
    beta2 = N[1] * beta2
    coeff = tf.stack([alpha1 + alpha2 - beta1 - beta2, beta2 - alpha2, beta1 - alpha1, tf.zeros_like(beta1)], axis=-1)
    coeff = tf.reshape(coeff, [-1, 2, 2, 1])
    
    # for test
    s1 = tf.reduce_sum((lines[:, :2] - v - alpha1 * (v1 - v) - alpha2 * (v2 - v))**2)
    s2 = tf.reduce_sum((lines[:, 2:] - v - beta1 * (v1 - v) - beta2 * (v2 - v))**2)
    coeff = tf.Print(coeff, ["error", s1, s2])

    return index, coeff, v, v1, v2 
    

# TESTS

def test_get_rotation_matrices():
    theta = tf.placeholder(tf.float32)
    M = get_rotation_matrices(theta)
    sess = tf.InteractiveSession()
    print(sess.run(M, feed_dict={theta : [0, math.pi / 2, math.pi]}))


def test_get_theta_by_lines():
    lines = tf.placeholder(tf.float32)
    theta = get_theta_by_lines(lines, -math.pi / 2)

    sess = tf.InteractiveSession()
    print(sess.run(theta, feed_dict={lines : [[0, 0, 1, 1], [0, 0, 0, 1]]}))


def test_get_quad_index_basis_by_lines():
    from grid import get_mesh_grid, fit_lines_to_grid_tf
    from quad import grid_to_quads
    N = [9, 10]
    R, C = get_mesh_grid(N) 
    quads = grid_to_quads(R, C) 
    lines = tf.constant([[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.1, 0.5], [0.5, 0.7, 0.1, 0.9], [0.4, 0.1, 0.5, 0.7]], dtype=tf.float32)
    lines = fit_lines_to_grid_tf(lines, N)
    index, coeff, v, v1, v2 = get_quad_index_basis_by_lines(lines, N)
    print("quads, index", quads, index)
    quads_l = tf.gather(quads, index)

    sess = tf.InteractiveSession()
    _lines, _quads_l, _coeff, _v, _v1, _v2 = sess.run([lines, quads_l, coeff, v, v1, v2])
    for i in range(_lines.shape[0]):
        print(_lines[i])
        print(_lines[i, 2:] - _lines[i, :2])
        print(np.sum(_coeff[i] * _quads_l[i], axis=(0, 1)))
        print("quad", _quads_l[i])
        print("v, v1, v2", _v[i], _v1[i], _v2[i])
        print(_coeff[i])
        print('----')


if __name__ == '__main__':
    #test_get_rotation_matrices()
    #test_get_theta_by_lines()
    test_get_quad_index_basis_by_lines()
