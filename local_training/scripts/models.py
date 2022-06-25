import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
from scripts.rigid_transform import regular_grid_2d, grid_transform, grid_sample_2d

### Several function are derived from the notebook affine-2d :
### https://colab.research.google.com/drive/1dRp2Ny2tH-NXddkT4pEzN6mtjEhnFCCw?usp=sharing#scrollTo=MMwa72GbVr1X.
### https://github.com/sarathknv/gsoc2020/blob/master/blogs/Deep-learning-based%202D%20and%203D%20Affine%20Registration.md
### created by sarathchandra.knv31@gmail.com 


@tf.function
def train_step(model,moving,static,criterion,optimizer):
    """ Make a single training step. 
    
    Make a single training step for a batch of pairs of 
    static and moving images. It makes a forward and 
    backward pass to compute the losses that it outputs.
    This function is based on the work of 
    sarathchandra.knv31@gmail.com in the notebook 
    https://colab.research.google.com/drive/1dRp2Ny2tH-NXddkT4pEzN6mtjEhnFCCw?usp=sharing#scrollTo=MMwa72GbVr1X
    
    """
    num_batch, W, H, C = tf.keras.backend.int_shape(moving)  # moving.shape
    # Define Gradient Tape
    with tf.GradientTape() as tape:
        # Get transformation matrices M
        #print(moving.shape, static.shape, "train")
        output_net = model({'moving': moving, 'static': static})
        # Compute the loss
        loss = criterion(static, moving, output_net)
    # Compute gradients.
    grads = tape.gradient(loss, model.trainable_variables)
    # Update the trainable parameters.
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

def deep_net(input_shape=(32,32,1)):
    """ Creates a Deep learning network for image registration. 
    
    The model is composed of convolutionnal layers to extract
    features in the images, and then dense layers to perform
    regression of the rigid transformations parameters. It is 
    based on the paper : 
    J. M. Sloan., K. A. Goatman., and J. P. Siebert., “Learning rigid
    image registration - utilizing convolutional neural networks for
    medical image registration,” in Proceedings of the 11th Inter-
    national Joint Conference on Biomedical Engineering Systems
    and Technologies - BIOIMAGING,, INSTICC. SciTePress,
    2018, pp. 89–99.

    """
    moving = layers.Input(shape=input_shape, name='moving')
    static = layers.Input(shape=input_shape, name='static')
    # Vision Tower Moving
    x_moving_skip = layers.Conv2D(64,kernel_size=3,strides=1,padding="same",activation="relu")(moving)
    x_moving = layers.Conv2D(64,kernel_size=3,strides=1,padding="same",activation="relu")(x_moving_skip)
    x_moving = layers.Add()([x_moving_skip,x_moving]) # skip connection
    # Vision Tower Static
    x_static_skip = layers.Conv2D(64,kernel_size=3,strides=1,padding="same",activation="relu")(static)
    x_static = layers.Conv2D(64,kernel_size=3,strides=1,padding="same",activation="relu")(x_static_skip)
    x_static = layers.Add()([x_static_skip,x_static]) # skip connection
    print(f"Shape of output of vision towers : {x_static.shape},{x_moving.shape}")
    # Merge both inputs into 1D array
    x = layers.concatenate([x_moving,x_static],axis=-1)
    x = layers.Flatten()(x)
    print(f"Shape of flatten input : {x.shape}")
    # Dropout to add regularization
    x = layers.Dropout(rate=0.5)(x)
    # Dense layers
    x = layers.Dense(units=64,activation="relu")(x)
    x = layers.Dense(units=64,activation="relu")(x)
    x = layers.Dense(units=3,activation="relu")(x)
    print(f"Neural Net output shape : {x.shape}") 
    # Final Model
    model = tf.keras.Model(inputs=[static,moving],outputs=x,name="Simple_DeepNet")
    return model

def fc_net(input_shape=(32,32,1)):
    """ Creates a Deep learning network for image registration. 
    
    The model is a fully convolutionnal network whose task is to 
    learn the transformations parameters for the registration. It
    is based on the paper : 
    J. M. Sloan., K. A. Goatman., and J. P. Siebert., “Learning rigid
    image registration - utilizing convolutional neural networks for
    medical image registration,” in Proceedings of the 11th Inter-
    national Joint Conference on Biomedical Engineering Systems
    and Technologies - BIOIMAGING,, INSTICC. SciTePress,
    2018, pp. 89–99.

    """
    moving = layers.Input(shape=input_shape, name='moving')
    static = layers.Input(shape=input_shape, name='static')
    # Vision Tower Moving
    x_moving_skip = layers.Conv2D(1,kernel_size=5,strides=1,padding="same",activation="relu")(moving)
    x_moving = layers.Conv2D(1,kernel_size=5,strides=1,padding="same",activation="relu")(x_moving_skip)
    x_moving = layers.concatenate([x_moving_skip,x_moving],axis=-1) # skip connection
    # Vision Tower Static
    x_static_skip = layers.Conv2D(1,kernel_size=5,strides=1,padding="same",activation="relu")(static)
    x_static = layers.Conv2D(1,kernel_size=5,strides=1,padding="same",activation="relu")(x_static_skip)
    x_static = layers.concatenate([x_static_skip,x_static],axis=-1) # skip connection
    print(f"Shape of output of vision towers : {x_static.shape},{x_moving.shape}")
    # Concatenate Towers
    x = layers.concatenate([x_static,x_moving],axis=-1)
    print(f"Shape of input for FCN : {x.shape}")
    # Fully Convolutional Layers
    x = layers.Conv2D(7,kernel_size=5,strides=2,padding="same",activation="relu")(x)
    x = layers.Conv2D(7,kernel_size=5,strides=2,padding="same",activation="relu")(x)
    x = layers.Conv2D(7,kernel_size=5,strides=2,padding="same",activation="relu")(x)
    x = layers.Conv2D(7,kernel_size=5,strides=2,padding="same",activation="relu")(x)
    x = layers.Conv2D(7,kernel_size=5,strides=2,padding="same",activation="relu")(x)
    x = layers.Conv2D(7,kernel_size=5,strides=2,padding="same",activation="relu")(x)
    x = layers.Conv2D(7,kernel_size=3,strides=2,padding="same",activation="relu")(x)
    x = layers.Conv2D(3,kernel_size=3,strides=2,padding="same",activation="linear")(x)
    x = layers.Conv2D(3,kernel_size=3,strides=2,padding="same",activation="linear")(x)
    print(f"Final Shape of network : {x.shape}")
    # Rotation matrix
    theta = x[:,0,0,0]
    rotation_matrix = tf.stack([tf.cos(theta),
                              -tf.sin(theta),  
                               tf.sin(theta),
                               tf.cos(theta)],axis=1)
    rotation_matrix = tf.reshape(rotation_matrix, (-1,2,2))
    print(f"Shape of rotation matrix {rotation_matrix.shape}")
    translation_vectors = x[:,0,0,1:]
    M = tf.concat([rotation_matrix,translation_vectors[:,:,np.newaxis]],axis=2)
    print(f"Final tranformation matrix shape : {M.shape}")
    # Move the image according to learned transformation
    grid = regular_grid_2d(input_shape[0],input_shape[1])
    grid_new = grid_transform(M, grid)
    grid_new = tf.clip_by_value(grid_new, -1, 1)
    moved = grid_sample_2d(moving, grid_new)
    # Final Model
    model = tf.keras.Model(inputs=[static,moving],outputs=moved,name="Simple_FCN")
    return model

def simple_cnn(input_shape=(32, 32, 1)):
    """ Creates a Deep learning network for image registration. 
    
    The model is composed of convolutionnal layers to extract
    features in the images, and then dense layers to perform
    regression of the rigid transformations parameters. It is 
    based on the work of sarathchandra.knv31@gmail.com in the 
    notebook :  
    https://colab.research.google.com/drive/1dRp2Ny2tH-NXddkT4pEzN6mtjEhnFCCw?usp=sharing#scrollTo=MMwa72GbVr1X.

    """
    moving = layers.Input(shape=input_shape, name='moving')
    static = layers.Input(shape=input_shape, name='static')
    x_in = layers.concatenate([static, moving], axis=-1)

    # Feature Exctraction
    x = layers.Conv2D(32, kernel_size=3, strides=2, padding='same',
                      activation='relu')(x_in)            
    x = layers.BatchNormalization()(x)                      
    x = layers.Conv2D(32, kernel_size=3, strides=1, padding='same',
                      activation='relu')(x)                 
    x = layers.BatchNormalization()(x)                      
    x = layers.MaxPool2D(pool_size=2)(x)                    
    x = layers.Conv2D(32, kernel_size=3, strides=1, padding='same',
                      activation='relu')(x)                 
    x = layers.BatchNormalization()(x)                      
    x = layers.Conv2D(32, kernel_size=3, strides=1, padding='same',
                      activation='relu')(x)                 
    x = layers.BatchNormalization()(x)                      
    x = layers.MaxPool2D(pool_size=2)(x)                    
    x = layers.Conv2D(32, kernel_size=3, strides=1, padding='same',
                      activation='relu')(x)                 
    x = layers.BatchNormalization()(x)                      
    # Parameters Regression
    x = layers.Flatten()(x)
    x = layers.Dense(32, activation='relu')(x)  
    x = layers.Dense(3, kernel_initializer='zeros')(x)
    print(f"Neural Net output shape : {x.shape}")
    
    # Final Model
    model = tf.keras.Model(inputs=[static,moving],outputs=x,name="Simple_DeepNet")
    return model

def CNN_layer(x,filter_nb,kernel_dim,max_pooling=True):
    """ Single CNN layer for the creation of more complex models. """
    x = layers.Conv2D(filter_nb, kernel_size=kernel_dim, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    if max_pooling:
        x = layers.MaxPool2D(pool_size=2,strides=2,padding="valid")(x)
    return x

def tower_extended(x):
    """ Vision tower used to extract features from the data. """
    x = CNN_layer(x,32,3)
    x = CNN_layer(x,64,3)
    x = CNN_layer(x,128,3,max_pooling=False)
    x = CNN_layer(x,64,1,max_pooling=False)
    x = CNN_layer(x,128,3)
    x = CNN_layer(x,256,3,max_pooling=False)
    x = CNN_layer(x,128,1,max_pooling=False)
    x = CNN_layer(x,256,3)
    x = CNN_layer(x,512,3,max_pooling=False)
    x = CNN_layer(x,256,1,max_pooling=False)
    x = CNN_layer(x,512,3,max_pooling=False)
    x = CNN_layer(x,256,1,max_pooling=False)
    x = CNN_layer(x,512,3)
    x = CNN_layer(x,1024,3,max_pooling=False)
    x = CNN_layer(x,512,1,max_pooling=False)
    x = CNN_layer(x,1024,3,max_pooling=False)
    x = CNN_layer(x,512,1,max_pooling=False)
    x = CNN_layer(x,1024,3,max_pooling=False)
    x = CNN_layer(x,2,1,max_pooling=False)
    print(x.shape)
    x = layers.GlobalAveragePooling2D()(x)
    return x

def model_extended(input_shape=(512,512,1)):
    """ Deep learning Network for registration. 
    
    The model is composed of a convolutionnal layers to extract
    features in the images, and then dense layers to perform
    regression of the rigid transformations parameters. It is 
    based on the paper :  
    K. T. Islam, S. Wijewickrema, and S. O’Leary, “A deep
    learning based framework for the registration of three
    dimensional multi-modal medical images of the head,”
    Scientific Reports, vol. 11, no. 1, Jan. 2021. [Online].
    Available: https://doi.org/10.1038/s41598-021-81044-7
    
    """
    moving = layers.Input(shape=input_shape, name='moving')
    static = layers.Input(shape=input_shape, name='static')
    # Extract image features
    moving_processed = tower_extended(moving)
    static_processed = tower_extended(static)
    # Perfom regression to find transformation parameters
    regression_input = layers.concatenate([moving_processed,static_processed],axis=-1)
    regression_input = layers.Flatten()(regression_input)
    reg = layers.Dense(10, activation='relu')(regression_input)
    reg_output = layers.Dense(3, kernel_initializer='zeros')(reg)
    # Final Model
    model = tf.keras.Model(inputs=[static,moving],outputs=reg_output,name="DeepNet_Extended")
    return model
