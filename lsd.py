import tensorflow as tf
from grid import fit_lines_to_grid, draw_grid, test_fit_line_to_grid, draw_line
import cv2
import numpy as np
# LSD

def lsd(image, N, fit=True):
    channels = image.get_shape().as_list()[-1]
    lsd = cv2.createLineSegmentDetector()
    
    def _lsd(image, N):
        lines, _, _, _ = lsd.detect(image)
        lines = np.squeeze(lines, axis=1)
        lines = np.float32(lines.reshape([-1, 4]))
        lines[:, 0:1] /= image.shape[0]
        lines[:, 2:3] /= image.shape[0]
        lines[:, 1:2] /= image.shape[1]
        lines[:, 3:4] /= image.shape[1]
        if fit:
            lines = fit_lines_to_grid(lines, N)
        return np.float32(lines)

    if channels == 3:
        image = tf.image.rgb_to_grayscale(image)
    lines = tf.py_func(_lsd, [image, N], tf.float32)
    lines.set_shape([None, 4])
    return lines

# TESTS

def test_lsd_equal(fit=True):
    import matplotlib.pyplot as plt
    image = tf.placeholder(tf.uint8, shape=[None, None, 3])
    N = [2, 3]
    path = '/media/george/1a4f4334-123f-430d-8a2b-f0c0fa401c75/data/alchohol/alchohol_104.jpeg'
    lines = lsd(image, N, fit=fit)
    sess = tf.InteractiveSession()

    _image = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    _image1 = draw_grid(_image.copy(), N)
    plt.imshow(_image1)
    plt.show()
    _lines = sess.run(lines, feed_dict={image : _image})
    _image1 = np.zeros_like(_image)
    for i in range(_lines.shape[0]):
        _image1 = draw_line(_image1, _lines[i])
    plt.imshow(_image1)
    plt.show()
    return _image1


def check_in_single_grid(line, N):
    eps = 1e-6
    def f(x, y, n):
        x = n * x
        y = n * y
        ix = int(x)
        iy = int(y)

        if ix == iy:
            return True
        if abs(ix - iy) > 2:
            return False

        if abs(ix - iy) == 2:
            return (abs(x - ix) < eps) and (abs(y - iy) < eps)
        
        return (abs(x - ix) < eps) or (abs(y - iy) < eps)
        
    return f(line[0], line[2], N[0]) and f(line[1], line[3], N[1])

def test_lsd_consistant(fit=True):
    import matplotlib.pyplot as plt
    image = tf.placeholder(tf.uint8, shape=[None, None, 3])
    N = [2, 3]
    path = '/media/george/1a4f4334-123f-430d-8a2b-f0c0fa401c75/data/alchohol/alchohol_104.jpeg'
    lines = lsd(image, N, fit=fit)
    sess = tf.InteractiveSession()

    _image = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    _image1 = draw_grid(_image.copy(), N)
    plt.imshow(_image1)
    plt.show()
    _lines = sess.run(lines, feed_dict={image : _image})
    for i in range(_lines.shape[0]):
        line = _lines[i]
        if not check_in_single_grid(line, N):
            print(line)


if __name__ == '__main__':
    '''
    a = test_lsd_equal()
    b = test_lsd_equal(False)
    print(np.sum((a - b)**2))
    '''
    test_lsd_consistant()
