import numpy as np
import os
import io
from skimage.color import rgb2gray
from skimage.filters import sato
from skimage import io
import skimage.feature
from tqdm import tqdm
import cv2
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
import json
import sys
import getopt

def main(argv):
    # Parse arguments
    try:
        opts, args = getopt.getopt(argv,"hv",["input_dir=","output_dir=","log_dir=","target_shape=",
                                 "max_angle=","max_translation=", "p_augment=", "model_path=",
                                "extra_input_dirs="])
    except getopt.GetoptError:
        print(argv)
        print("Error in inputs, please be sure to provide valid keywords (refer to help section)")
        sys.exit(1)
    input_dir = ""
    output_dir = ""
    log_dir = ""
    target_shape = (512,512)
    vessel = False
    max_angle = 15
    max_translation = 50
    p_augment = 0.1
    model_path = None
    extra_input_dirs = None
    for opt, arg in opts:
        if opt == "-h":
            print("\n")
            print("Help section:\n")
            print("Format and process the data in preparation for model training.\n")
            print("This pipeline will process the data in the given folder, merged with") 
            print("the other extra input folders if given, but keeping only shared patients.")
            print("The first processing is resizing to target shape with black padding to")
            print("preserve ratio and extract vessel binary masks. Then if needed it will")
            print("perform data augmentation, i.e. in the cases where there is a single image")
            print("for a given type so no registration is possible. It will also perform ")
            print("randomly some data augmentation to avoid bias. A json file will be")
            print("created with the mapping information between the old data set and the new one.")
            print("The format of this file can be looked at in the github : ...\n")
            print("Parameters:")
            print("-input_dir: (String) Path to the input directory.")
            print("-output_dir: (String) Path to the output directory (will be created).")
            print("-log_dir: (String) Path to the directory for the mapping file.\n")
            print("Keywords:")
            print("-target_shape: (int,int) Target shape for the processed images.")
            print("-vessel: (bool) Indicate if we want to use the segmentation deepnet or not.")
            print("-max_angle: (int) Maximum angle for data augmentation.")
            print("-max_translation: (int) Maximum x and y translation in data augmentation. ")
            print("-p_augment: (float) Probability of randomly augmenting valid data.")
            print("-model_path: (String) Path to the model for vessel segmentation, required if vessel is True. ")
            print("-extra_input_dirs: (String) Pathes to the other directories if multi folder")
            print("\t\t\t\t\t\tis wanted. Should be in format : path1,path2,...")
            sys.exit()
        elif opt == "-v":
            vessel = True
        elif opt == "--input_dir":
            input_dir = arg
        elif opt == "--output_dir":
            output_dir = arg
        elif opt == "--log_dir":
            log_dir = arg
        elif opt == "--model_path":
            model_path = arg
        elif opt == "--extra_input_dirs":
            extra_input_dirs = arg.split(",")
        elif opt == "--target_shape":
            width = int(arg.split(",")[0][1:])
            height = int(arg.split(",")[1][:-1])
            target_shape = (width,height)
        elif opt == "--max_angle":
            max_angle = int(arg)
        elif opt == "--max_translation":
            max_translation = int(arg)
        elif opt == "--p_augment":
            p_augment = int(arg)
    os.mkdir(output_dir)
    data_processing(input_dir, output_dir, log_dir, target_shape=target_shape, vessel=vessel,
                    max_angle=max_angle, max_translation=max_translation, p_augment=p_augment, 
                    model_path=model_path, extra_input_dirs=extra_input_dirs)

# Model metrics, used for vessel segmentation
# Taken from the work of N. Thomar, Unet segmentation in keras
# tensorflow, https://github.com/nikhilroxtomar/UNet-Segmentation-in-Keras-TensorFlow, 2020.

smooth = 1e-15

def iou(y_true, y_pred):
    """ Compute the IoU. """
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)

def dice_coef(y_true, y_pred):
    """ Compute the dice coefficient. """
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    """ Compute the dice loss. """
    return 1.0 - dice_coef(y_true, y_pred)

# Functions to create vessel masks using deeplearning or standard image processing

def increase_vessel_contrast(image,model,rate=0.25,only_vessel=False):
    """ Return the image with darker vessels
    
    Given an image it will use a pre-trained vessel segmentation
    model to detect the vessels in the eye. Then it will use this
    annotation as a mask to increase the darkness of vessels according 
    to the given rate. If only_vessel is given, then it will output 
    the vessel mask.
    
    image : (numpy ndarray) image to process
    rate : (float) scale the darkness of vessels
    model : (keras.engine.functional.Functional) Pre-trained 
                model for vessel segmentation
    only_vessel : (bool) indicate if only the mask should be outputed

    """
    x = image/255.0
    x = x.astype(np.float32)
    """ Prediction """
    y_pred = model.predict(np.expand_dims(x, axis=0))[0]
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.int32)
    y_pred = np.squeeze(y_pred, axis=-1)
    y_pred = np.expand_dims(y_pred, axis=-1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1) * 255
    if only_vessel:
        y_pred = y_pred.astype(np.uint8)
        return y_pred
    """ Enhance vessels """
    z = image
    z = z.astype(np.float32)
    z -= rate*y_pred
    z[z < 0] = 0
    z[z > 255] = 255
    return z

def segment_vessels(filename):
    """ Output a binary mask of the vessel in the given file. 
    
    Copyright Dherse Alexandre Pierre, alexandrepierre.dherse@fa2.ch
    
    """
    img = io.imread(filename)
    gray_img = rgb2gray(img)
    # The threshold is set empirically to have
    # fine grain annotation but not too much to avoid
    # noisy or too complex mask
    threshold = 0.01 
    ridges = sato(gray_img, mode="reflect")
    ridges[ridges >= threshold] = 255
    ridges[ridges < threshold] = 0
    # mask to remove border of image
    hh = int(ridges.shape[0] / 2)
    hw = int(ridges.shape[1] / 2)
    rr, cc = skimage.draw.disk((hh,hw), 0.9*min(hh,hw))
    mask = np.zeros(ridges.shape, dtype=np.uint8)
    mask[rr,cc] = 1
    masked_image = ridges * mask

    return masked_image.astype(np.uint8)

# Functions to extract and process the informations in the file names

def gather_by_patient(input_dir):
    """ Given a list of files gather them into a dictionnary per patient. """
    # We remove the file extension and append the file directory
    # at the end to keep track of its location. 
    files = [f+"-"+input_dir for f in os.listdir(input_dir)
                       if os.path.isfile(input_dir+"/"+f)]

    images_per_patient = {}
    for file_name in tqdm(files,total=len(files)):
        data = file_name.split("-")
        if len(data) > 8 or data[4] == "LQ": # avoid replicating outliers
            continue
        patient_id = data[0]
        if patient_id not in images_per_patient.keys():
            images_per_patient[patient_id] = []
        images_per_patient[patient_id] += [file_name]
    return images_per_patient

def merge_per_patient(images_1,images_2):
    """ Given two dictionnaries of images per patient, merge the common ones into a single dict. """
    merged_images = {}
    for patient_id in images_1.keys():
        if patient_id in images_2.keys():
            merged_images[patient_id] = images_1[patient_id] + images_2[patient_id]
    return merged_images

def sort_image_data(images):
    """ Given the data for several images, gather them by type in a dict. """
    # We split to have the info in the file name.
    images_data = [img.split("-") for img in images]
    data_dict = {"L-macula":[], "R-macula":[],
                "L-OHN":[], "R-OHN":[],
                "L-other":[], "R-other":[]}
    for image_data, image in zip(images_data,images):
        side = image_data[2]
        if len(image_data) < 8: # 7 is the standard size but we added the file name at the end
            data_dict[side+"-other"] += [image]
        else:
            centerdness = image_data[4]
            data_dict[side+"-"+centerdness] += [image]
    return data_dict

# Functions to process the images

def create_mask(image_path,target_shape,vessel,model=None):
    """ Create the mask need as model input from the given image. """
    if vessel:
        original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        good_shape_image = reshape_image(original_image.copy(),target_shape)
        mask = increase_vessel_contrast(good_shape_image.copy(),model,only_vessel=True)
    else:
        mask_image = segment_vessels(image_path)
        mask = reshape_image(mask_image.copy()[:,:,np.newaxis],target_shape)
    return mask

def augment_data(image,max_angle,max_translation):
    """ Given an image apply a random rigid transform. """
    # Get random transformations parameters
    angle = np.random.randint(low=-max_angle,high=max_angle)
    tx = np.random.randint(low=-max_translation,high=max_translation)
    ty = np.random.randint(low=-max_translation,high=max_translation)
    height, width = image.shape[:2]
    center_X = width // 2 # images are squares 
    center_Y = center_X
    # Apply the rotation
    M = cv2.getRotationMatrix2D((center_X, center_Y), angle, 1.0)
    rotated_image = cv2.warpAffine(image, M, (width, height))
    # Apply the translation
    M = np.float32([[1,0,tx],
                    [0,1,ty]])
    transformed_image = cv2.warpAffine(rotated_image, M, (height,width))
    return transformed_image, angle, tx, ty

def process_data(data,output_folder,target_shape,vessel,max_angle,
                 max_translation,p_augment, data_mapping, pair_id, model=None):
    """ Given some images, will extract their mask and augment them if needed. """
    masks = []
    if len(data) == 1:
        data_info = data[0].split("-")
        image_path = data_info[-1]+"/"+"-".join(data_info[:-1])
        original_mask = create_mask(image_path,target_shape,vessel,model=model)
        # Need to augment the data as we would not have enough images to do registration
        new_mask, angle, tx , ty = augment_data(original_mask,max_angle,max_translation)
        masks.append(original_mask)
        masks.append(new_mask)
        data_mapping[str(pair_id)+"/0"] = {"path":image_path, "augmented":"False", "params":"-/-/-"}
        data_mapping[str(pair_id)+"/1"] = {"path":image_path, "augmented":"True", 
                                           "params": str(angle)+"/"+str(tx)+"/"+str(ty)}
    else:
        idx = 0
        for d in data:
            data_info = d.split("-")
            image_path = data_info[-1]+"/"+"-".join(data_info[:-1])
            original_mask = create_mask(image_path,target_shape,vessel,model=model)
            masks.append(original_mask)
            data_mapping[str(pair_id)+"/"+str(idx)] = {"path":image_path, "augmented":"False",
                                                       "params":"-/-/-"}
            idx += 1
            # Add randomly augmented data to avoid bias compared to case with single image
            flip = np.random.binomial(1,p_augment)
            if flip:
                new_mask, angle, tx, ty = augment_data(original_mask,max_angle,max_translation)
                masks.append(new_mask)
                data_mapping[str(pair_id)+"/"+str(idx)] = {"path":image_path, "augmented":"True", 
                                                        "params": str(angle)+"/"+str(tx)+"/"+str(ty)}
                idx += 1
    return masks

def reshape_image(image,target_shape):
    """ Reshape the image to the target shape"""
    old_image_height, old_image_width, channels = image.shape
    # create square of black
    square_dim = max(old_image_height,old_image_width)
    square_padding = np.full((square_dim,square_dim, channels), fill_value=0, dtype=np.uint8)
    # compute centering coordinate
    x_center = (square_dim - old_image_width) // 2
    y_center = (square_dim - old_image_height) // 2
    # paste the image in our new square box
    square_padding[y_center:y_center+old_image_height, 
           x_center:x_center+old_image_width] = image
    # resize the square but preserve ratio
    new_image = cv2.resize(square_padding,(target_shape[0],target_shape[1])) 
    return new_image

def save_data(image, idx, output_path):
    """ Save the image with label idx in format jpg. """
    tmp = image.copy()
    tmp = tmp.astype(np.uint8)
    cv2.imwrite(output_path+"/"+str(idx)+".jpg",tmp)

# Final Pipeline

def data_processing(input_dir, output_dir, log_dir, target_shape=(512,512), vessel=True,
                    max_angle=15, max_translation=50, p_augment=0.1, 
                    model_path=None, extra_input_dirs=None):
    """ 
    Format and process the data in preparation for model training.

    This pipeline will process the data in the given folder, merged with 
    the other extra input folders if given, but keeping only shared patients.
    The first processing is resizing to target shape with black padding to
    preserve ratio and extract vessel binary masks. Then if needed it will 
    perform data augmentation, i.e. in the cases where there is a single image
    for a given type so no registration is possible. It will also perform 
    randomly some data augmentation to avoid bias. A json file will be
    created with the mapping information between the old data set and the new one.
    The format of this file can be looked at in the github : ...

    Parameters:
    -input_dir: (String) Path to the input directory.
    -output_dir: (String) Path to the output directory (will be created).
    -log_dir: (String) Path to the directory for the mapping file.

    Keywords:
    -target_shape: (int,int) Target shape for the processed images.
    -vessel: (bool) Indicate if we want to use the segmentation deepnet or not.
    -max_angle: (int) Maximum angle for data augmentation.
    -max_translation: (int) Maximum x and y translation in data augmentation. 
    -p_augment: (float) Probability of randomly augmenting valid data.
    -model_path: (String) Path to the model for vessel segmentation, required if vessel is True. 
    -extra_input_dirs: (String) Pathes to the other directories if multi folder
                            is wanted. Should be in format : path1,path2,...

    """
    print("Gather the images per patient")
    images_per_patient = gather_by_patient(input_dir)
    # In the case we want to aggregate data from multiple folders.
    if extra_input_dirs:
        for extra_dir in extra_input_dirs:
            print(f"Gathering images from extra directory {extra_dir}")
            extra_images_per_patient = gather_by_patient(extra_dir)
            images_per_patient = merge_per_patient(images_per_patient,
                                                   extra_images_per_patient)
    print(f"We have now {len(images_per_patient.keys())} patients")
    print("Begin data processing")
    model = None
    if model_path:
        print(f"Load vessel segmentation model located in {model_path}")
        with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
            model = tf.keras.models.load_model(model_path)
        print("Model loaded")
    pair_id = 0
    data_mapping = {}
    for patient in tqdm(images_per_patient.keys(), total=len(images_per_patient.keys())):
        images = images_per_patient[patient]
        # We sort the files to create folder for each compatible registrations class
        sorted_images_data = sort_image_data(images)
        for data in sorted_images_data.values(): 
            if len(data) > 0:
                data_path = output_dir+"/"+str(pair_id)
                os.mkdir(data_path)
                masks = process_data(data,data_path,target_shape,vessel,
                                     max_angle,max_translation,p_augment,
                                     data_mapping, pair_id, model=model)
                for idx, m in enumerate(masks):
                    save_data(m, idx, data_path)
                pair_id += 1
    with open(log_dir+"/mapping_info.json","w") as output:
        json.dump(data_mapping, output)
    print("Data Processing executed with success!")
    print(f"The processed images are located in {output_dir}")

if __name__ == "__main__":
    main(sys.argv[1:])