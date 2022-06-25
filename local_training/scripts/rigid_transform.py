#MIT License

#Copyright (c) 2017 Kevin Zakka

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

### Based on work from sarathchandra.knv31@gmail.com which
### is based on https://kevinzakka.github.io/2017/01/10/stn-part1/
### and https://github.com/sarathknv/gsoc2020/blob/master/blogs/Deep-learning-based%202D%20and%203D%20Affine%20Registration.md

import tensorflow as tf
import numpy as np

def apply_transformation_matrix(M, moving):
    """ Move the image according to the given transformation. """
    input_shape = tf.shape(moving[0])
    # Move the image according to learned transformation
    grid = regular_grid_2d(input_shape[0],input_shape[1])
    grid_new = grid_transform(M, grid)
    grid_new = tf.clip_by_value(grid_new, -1, 1)
    return grid_sample_2d(moving, grid_new)

def build_transformation_matrix(output_net):
    """ Create the transformation matrix based on given parameters. """
    theta = output_net[:,0]
    rotation_matrix = tf.stack([tf.cos(theta),
                              -tf.sin(theta),  
                               tf.sin(theta),
                               tf.cos(theta)],axis=1)
    rotation_matrix = tf.reshape(rotation_matrix, (-1,2,2))
    #print(f"Shape of rotation matrix {rotation_matrix.shape}")
    translation_vectors = output_net[:,1:]
    M = tf.concat([rotation_matrix,translation_vectors[:,:,np.newaxis]],axis=2)
    #print(f"Final tranformation matrix shape : {M.shape}")
    return M

def regular_grid_2d(height, width):
    """ Create a normalized 2d grid. """
    x = tf.linspace(-1.0, 1.0, width)  # shape (W, )
    y = tf.linspace(-1.0, 1.0, height)  # shape (H, )
    X, Y = tf.meshgrid(x, y)  # shape (H, W), both X and Y
    grid = tf.stack([X, Y], axis=-1)
    return grid

def grid_transform(theta, grid):
    """ Apply the given transformation theta to the grid. """
    nb = tf.shape(theta)[0]
    nh = tf.shape(grid)[0]
    nw = tf.shape(grid)[1]
    
    x = grid[..., 0]  
    y = grid[..., 1]

    x_flat = tf.reshape(x, shape=[-1]) 
    y_flat = tf.reshape(y, shape=[-1])
    # reshape to (xt, yt, 1) to allow translation
    ones = tf.ones_like(x_flat)
    grid_flat = tf.stack([x_flat, y_flat, ones])
    grid_flat = tf.expand_dims(grid_flat, axis=0)
    # repeat grid num_batch times
    grid_flat = tf.tile(grid_flat, tf.stack([nb, 1, 1]))

    theta = tf.cast(theta, 'float32')
    grid_flat = tf.cast(grid_flat, 'float32')

    # transform the sampling grid i.e. batch multiply
    grid_new = tf.matmul(theta, grid_flat)  # n, 2, h*w
    # reshape to (num_batch, height, width, 2)
    grid_new = tf.transpose(grid_new, perm=[0,2,1])
    grid_new = tf.reshape(grid_new, [nb, nh, nw, 2])

    return grid_new

def grid_sample_2d(moving, grid):
    """ Sample the moving image according to the given grid. """
    #nb, nh, nw, nc = tf.shape(moving)
    nb = tf.shape(moving)[0]
    nh = tf.shape(moving)[1]
    nw = tf.shape(moving)[2]
    nc = tf.shape(moving)[3]

    x = grid[..., 0]  # shape (N, H, W)
    y = grid[..., 1]
    x = tf.cast(x, 'float32')
    y = tf.cast(y, 'float32')

    # Scale x and y from [-1.0, 1.0] to [0, W] and [0, H] respectively.
    x = (x + 1.0) * 0.5 * tf.cast(nw-1, 'float32')
    y = (y + 1.0) * 0.5 * tf.cast(nh-1, 'float32')

    y_max = tf.cast(nh - 1, 'int32')
    x_max = tf.cast(nw - 1, 'int32')
    zero = tf.constant(0, 'int32')

     # grab 4 nearest corner points for each (x_i, y_i).
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1

    # make sure it's inside img range [0, H] or [0, W]
    x0 = tf.clip_by_value(x0, zero, x_max)
    x1 = tf.clip_by_value(x1, zero, x_max)
    y0 = tf.clip_by_value(y0, zero, y_max)
    y1 = tf.clip_by_value(y1, zero, y_max)

    # Collect indices of the four corners.
    b = tf.ones_like(x0) * tf.reshape(tf.range(nb), [nb, 1, 1])
    idx_a = tf.stack([b, y0, x0], axis=-1)  # all top-left corners
    idx_b = tf.stack([b, y1, x0], axis=-1)  # all bottom-left corners
    idx_c = tf.stack([b, y0, x1], axis=-1)  # all top-right corners
    idx_d = tf.stack([b, y1, x1], axis=-1)  # all bottom-right corners
    # shape (N, H, W, 3)

    # look up pixel values at corner coords
    moving_a = tf.gather_nd(moving, idx_a)  # all top-left values
    moving_b = tf.gather_nd(moving, idx_b)  # all bottom-left values
    moving_c = tf.gather_nd(moving, idx_c)  # all top-right values
    moving_d = tf.gather_nd(moving, idx_d)  # all bottom-right values
    # shape (N, H, W, C)
    
    x0_f = tf.cast(x0, 'float32')
    x1_f = tf.cast(x1, 'float32')
    y0_f = tf.cast(y0, 'float32')
    y1_f = tf.cast(y1, 'float32')

    # calculate deltas and add dimension for addition
    wa = tf.expand_dims((x1_f - x) * (y1_f - y), axis=-1)
    wb = tf.expand_dims((x1_f - x) * (y - y0_f), axis=-1)
    wc = tf.expand_dims((x - x0_f) * (y1_f - y), axis=-1)
    wd = tf.expand_dims((x - x0_f) * (y - y0_f), axis=-1)
    
    # Calculate the weighted sum.
    moved = tf.add_n([tf.multiply(wa,moving_a), tf.multiply(wb,moving_b), tf.multiply(wc,moving_c),
                      tf.multiply(wd,moving_d)])
    return moved