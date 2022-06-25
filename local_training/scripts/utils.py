import cv2
import numpy as np
import os 
import tensorflow as tf
import matplotlib.pyplot as plt

def load_masks(input_path,target_size=(512,512)):
    """ Dataloader that load every possible registration combination in the given path. """
    patients_nb = len(os.listdir(input_path))
    images_per_patients = len(os.listdir(input_path+"/"+os.listdir(input_path)[0]))
    tot = patients_nb * images_per_patients * (images_per_patients-1)
    data = np.zeros((tot,2,target_size[0],target_size[1],1),dtype=np.float32)
    for i in range(patients_nb):
        for j in range(images_per_patients):
            index = i*images_per_patients*(images_per_patients-1)+j*(images_per_patients-1)
            for k in range(images_per_patients):
                if k != j:
                    fix = cv2.imread(input_path+"/"+str(i)+"/"+str(j)+".jpg",cv2.IMREAD_GRAYSCALE)
                    moved = cv2.imread(input_path+"/"+str(i)+"/"+str(k)+".jpg",cv2.IMREAD_GRAYSCALE)
                    fix = cv2.resize(fix,target_size)
                    moved = cv2.resize(moved,target_size)
                    data[index][0] = np.asarray(fix)[:,:,np.newaxis]/255
                    data[index][1] = np.asarray(moved)[:,:,np.newaxis]/255
                    index += 1
    return data

def split_dataset(data, balance=0.7, seed=10):
    """ Split the given dataset in a train and test set in tensor format. """
    np.random.seed(seed)
    np.random.shuffle(data)
    threshold = int(data.shape[0] * balance)
    train = data[:threshold]
    test = data[threshold:]
    train = tf.convert_to_tensor(train)
    test = tf.convert_to_tensor(test)
    return train, test

def show_images(images):
    """ Takes a list of images and output a horizontal concatenated version of them. """
    full_image = np.concatenate(images, axis=1)
    cv2.imshow("Full_Image",full_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def plot_images(moved, moving, static):
    """ Plot the two initial images and the output of the registration. 
    
    Derived from the affine-2d code of @sarathchandra.knv31@gmail.com
    """
    nb, _, _, _ = moving.shape

    # Convert back tensors to 8-bit images.
    moved = moved.numpy()[0,:,:,:]*255 # moved.numpy().squeeze(axis=-1) * 255.0
    moved = moved.astype(np.uint8)
    moving = moving[0,:,:,:]*255 # moving.numpy().squeeze(axis=-1) * 255.0
    moving = moving.astype(np.uint8)
    static = static[0,:,:,:]*255 # static.numpy().squeeze(axis=-1) * 255.0
    static = static.astype(np.uint8)

    # Plot contiguously the images.
    # Order: Moved, Static, Moving
    cv2.imshow('image window', np.concatenate([moved,static, moving],axis=0))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def plot_images_2(model, moving, static):
    nb, nh, nw, nc = moving.shape

    # Repeat the static image along the batch dim.
    multiples = tf.constant([nb, 1, 1, 1], tf.int32)
    static = tf.tile(static, multiples)

    moved = model({'moving': moving, 'static': static}, training=False)

    # Convert the tensors to 8-bit images.
    moved = moved.numpy().squeeze(axis=-1) * 255.0
    moved = moved.astype(np.uint8)
    moving = moving.numpy().squeeze(axis=-1) * 255.0
    moving = moving.astype(np.uint8)
    static = static.numpy().squeeze(axis=-1) * 255.0
    static = static.astype(np.uint8)

    # Plot images.
    fig = plt.figure(figsize=(3 * 3.7, nb * 3.7))
    titles_list = ['Static', 'Moved', 'Moving']
    images_list = [static, moved, moving]
    for i in range(nb):
        for j in range(3):
            ax = fig.add_subplot(nb, 3, i * 3 + j + 1)
            if i == 0:
                ax.set_title(titles_list[j], fontsize=20)
            ax.set_axis_off()
            ax.imshow(images_list[j][i], cmap='gray')

    plt.tight_layout()
    plt.show()