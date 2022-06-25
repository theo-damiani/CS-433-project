import tensorflow as tf
from rigid_transform import build_transformation_matrix, apply_transformation_matrix

### These functions are derived from the work of sarathchandra.knv31@gmail.com
### and its notebook affine-2d : https://colab.research.google.com/drive/1dRp2Ny2tH-NXddkT4pEzN6mtjEhnFCCw?usp=sharing#scrollTo=MMwa72GbVr1X
### https://github.com/sarathknv/gsoc2020/blob/master/blogs/Deep-learning-based%202D%20and%203D%20Affine%20Registration.md

@tf.function
def mse_loss_reg(static, moving, output_net):
    """ Mean Square error between the static and moved images with regularization on transform parameters. """  
    M = build_transformation_matrix(output_net)
    # Move the image according to learned transformation    
    moved = apply_transformation_matrix(M, moving)
    
    # Compute the losses
    loss = tf.reduce_mean(tf.square(static - moved)) 
    loss2 = tf.reduce_mean(tf.square(moving - moved))  
    reg = tf.reduce_mean(tf.square(output_net))
    
    reg_factor = 0.01
    return loss + loss2 + reg_factor*reg

@tf.function
def mse_loss(static, moving, output_net):
    """ Mean Square error between the static and moved images. """ 
    M = build_transformation_matrix(output_net)
    # Move the image according to learned transformation    
    moved = apply_transformation_matrix(M, moving)
    # Comput the loss
    loss = tf.reduce_mean(tf.square(moved - static))  
    return loss

@tf.function
def ncc_loss(static, moving, output_net):
    """ Normalized Cross Correlations between the two static and moved images. """

    M = build_transformation_matrix(output_net)
    # Move the image according to learned transformation    
    moved = apply_transformation_matrix(M, moving)
    
    eps = tf.constant(1e-9, 'float32')

    static_mean = tf.reduce_mean(static, axis=[1, 2], keepdims=True)
    moved_mean = tf.reduce_mean(moved, axis=[1, 2], keepdims=True)
    # shape (N, 1, 1, C)

    static_std = tf.math.reduce_std(static, axis=[1, 2], keepdims=True)
    moved_std = tf.math.reduce_std(moved, axis=[1, 2], keepdims=True)
    # shape (N, 1, 1, C)

    static_hat = (static - static_mean)/(static_std + eps)
    moved_hat = (moved - moved_mean)/(moved_std + eps)
    # shape (N, H, W, C)

    ncc = tf.reduce_mean(static_hat * moved_hat)  # shape ()
    loss = -ncc
    return loss