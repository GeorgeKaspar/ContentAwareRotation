import tensorflow as tf
from grid import get_mesh_grid 

# QUADS

def grid_to_quads(R, C):
    '''
    first, second - row, col
    quad: [[p00, p01], [p10, p11]]
    quads: q00 q01 q02 ... q10 q12 ... ... qn0 ... qnm

    '''
    def extract_patches(x):
        shape = x.get_shape().as_list()
        print("shape", shape)
        x = tf.expand_dims(x, axis=0)
        x = tf.expand_dims(x, axis=3)
        x = tf.extract_image_patches(x, [1,2,2,1], [1,1,1,1], [1,1,1,1], 'VALID')
        x = tf.reshape(x, [shape[0] - 1, shape[1] - 1, 2, 2])
        x = tf.transpose(x, [1, 0, 2, 3])
        x = tf.reshape(x, [-1, 2, 2])
        return x
    print(R.shape, C.shape)
    R_patches = extract_patches(R)
    C_patches = extract_patches(C)
    quads = tf.transpose(tf.stack([R_patches, C_patches], axis=-1), [0, 2, 1, 3])
    print(quads.shape)
    return quads

def get_Aq(R, C):
    z = tf.constant([[-1, 1, -1, 1, -1, 1, -1, 1]], dtype=tf.float32)
    a = tf.constant([1, 0, 1, 0, 1, 0, 1, 0], dtype=tf.float32)
    b = tf.constant([0, 1, 0, 1, 0, 1, 0, 1], dtype=tf.float32)
    quads = grid_to_quads(R, C)
    num_quads = tf.shape(quads)[0:1]
    xy = tf.manip.roll(quads, shift=1, axis=3)
    yx = quads
    col1 = tf.reshape(xy, (-1, 8))
    col2 = tf.reshape(yx, (-1, 8)) * z
    col3 = tf.reshape(tf.tile(a, num_quads), (-1, 8))
    col4 = tf.reshape(tf.tile(b, num_quads), (-1, 8))
    Aq = tf.stack([col1, col2, col3, col4], axis=2)
    return Aq

def get_Vq(R, C):
    quads = grid_to_quads(R, C)
    xy = tf.manip.roll(quads, shift=1, axis=3)
    return tf.reshape(xy, (-1, 8, 1))
# TEST

def test_grid_to_quads():
    N = [2, 3]
    R, C = get_mesh_grid(N)
    patches = grid_to_quads(R, C)
    sess = tf.InteractiveSession()
    print(N)
    print(sess.run([R, C]))
    print(sess.run(patches))

def test_Aq():
    N = [3, 3]
    R, C = get_mesh_grid(N)
    Aq = get_Aq(R, C) 
    sess = tf.InteractiveSession()
    print(sess.run(Aq))

def test_Vq():
    N = [3, 3]
    R, C = get_mesh_grid(N)
    Vq = get_Vq(R, C) 
    sess = tf.InteractiveSession()
    print(sess.run(Vq))

if __name__ == '__main__':
    test_grid_to_quads()
    #test_Vq()

