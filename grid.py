import tensorflow as tf
import numpy as np
import math
import cv2

# FIT TO GRID

def fit_line_to_grid(p, N):
    '''
    p[0] - point1
    p[1] - point2
    '''
    res = [0, 1]
    v = p[1] - p[0]
    for axis in [0, 1]:
        a = N[axis] * p[0, axis]
        b = N[axis] * p[1, axis]
        for k in range(math.ceil(min(a, b)), math.floor(max(a, b)) + 1):
            k = float(k)
            t = ((k / N[axis]) - p[0, axis]) / v[axis]
            res.append(t)
    res = np.reshape(np.unique(res), [-1, 1])
    p = p[0] + v * res
    return np.float32(np.concatenate([p[:-1, :], p[1:, :]], axis=-1))

def fit_lines_to_grid(lines, N):
    res = []
    for idx in range(lines.shape[0]):
        line = lines[idx]
        #print("in", line)
        res.append(fit_line_to_grid(np.reshape(line, (2, 2)), N))
        #print("out", res[-1])
        #assert len(res[-1]) > 0, "Empty"
    return np.concatenate(res, axis=0)

def fit_lines_to_grid_tf(lines, N):
    lines = tf.py_func(fit_lines_to_grid, [lines, N], tf.float32)
    lines.set_shape([None, 4])
    return lines

# DRAW

def draw_grid(image, N):
    if len(image.shape) == 2:
        for i in range(N[0] + 1):   
            r = min(i * image.shape[0] // N[0], image.shape[0] - 1)
            image[r, :] = 255

        for i in range(N[1] + 1):
            c = min(i * image.shape[1] // N[1], image.shape[1] - 1)
            image[:, c] = 255
    else:
        for i in range(N[0] + 1):   
            r = min(i * image.shape[0] // N[0], image.shape[0] - 1)
            image[r, :, :] = [255, 0, 0]

        for i in range(N[1] + 1):
            c = min(i * image.shape[1] // N[1], image.shape[1] - 1)
            image[:, c, :] = [255, 0, 0]

    return image

def draw_line(img, line):
    shape = img.shape
    p1 = (np.int64(line[0] * shape[0]), np.int64(line[1] * shape[1]))
    p2 = (np.int64(line[2] * shape[0]), np.int64(line[3] * shape[1]))
    return cv2.line(img, p1, p2, [255])
    
# MAKE grid

def get_mesh_grid(N):
    zero32 = tf.constant(0, dtype=tf.float32)
    one32 = tf.constant(1, dtype=tf.float32)
    r = tf.linspace(zero32, one32, N[0] + 1)
    c = tf.linspace(zero32, one32, N[1] + 1)
    R, C = tf.meshgrid(r, c)
    return R, C

# TESTS

def test_fit_line_to_grid(p, N):
    import cv2
    import matplotlib.pyplot as plt
    shape = (512, 680)
    img = np.zeros(shape)
    for line in p:
        p1 = (np.int64(line[1] * shape[1]), np.int64(line[0] * shape[0]))
        p2 = (np.int64(line[3] * shape[1]), np.int64(line[2] * shape[0]))
        img = cv2.line(img, p1, p2, [255])
 
    plt.imshow(img)
    plt.show() 
    
    img = np.zeros(shape)
    draw_grid(img, N)
    plt.imshow(img)
    plt.show()
    print(p)
    res = fit_lines_to_grid(p, N)
    print(res)
    for line in res:
        p1 = (np.int64(line[1] * shape[1]), np.int64(line[0] * shape[0]))
        p2 = (np.int64(line[3] * shape[1]), np.int64(line[2] * shape[0]))
        img = cv2.line(img, p1, p2, [255])
 

        plt.imshow(img)
        plt.show()
    return img

def test_mesh_grid():
    image = tf.placeholder(tf.float32, shape=[100, 200, 3])
    N = [4, 4]
    R, C = get_mesh_grid(N)
    sess = tf.InteractiveSession()
    _R, _C = sess.run([R, C], feed_dict={image : np.zeros((100, 200, 3))})
    print(_R)
    print(_C)



if __name__ == '__main__':
    #p = np.array([[0.9, 0.9], [0.3, 0.4]])
    #p = np.array([[0.7163304, 0.99734706, 0.7006974, 0.9628671], [0.2, 0.4, 0.5, 0.21], [0.1111, 0.011, 0.9111, 0.81111111], [0.1, 0.11, 0.113, 0.114]])
    #test_fit_line_to_grid(p, [4, 4])
    test_mesh_grid()

