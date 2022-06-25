from scripts.data_processing import *
from scripts.rigid_transform import *
from scripts.utils import *
from scripts.loss import *
import sys
import getopt
import json
import cv2
from tensorflow import keras
from skimage.color import rgb2gray


def main(argv):
    # Parse arguments
    try:
        opts, args = getopt.getopt(argv,"hs",["input_file=","output_dir=",
                                 "max_angle=","max_translation=", "model_path=", "metric_path="])
    except getopt.GetoptError:
        print(argv)
        print("Error in inputs, please be sure to provide valid keywords (refer to help section)")
        sys.exit(1)
    input_file = ""
    output_dir = ""
    model_path = ""
    show = False
    vessel = False
    seg_model = None
    target_shape = (512,512)
    max_angle = 7
    max_translation = 15
    metric_path = "ncc.json"
    metric_path = ""
    for opt, arg in opts:
        if opt == "-h":
            print("\n")
            print("Help section:\n")
            print("Provided an input file, this script will make the rigid registration.")
            print("of all image pairs provided.")
            print("It will also output a file with the metrics (ncc) between the registered")
            print("image and the static one. Check the github to see an example of input file.")
            print("The order of the images is moved, static, moving.\n")
            print("Parameters:\n")
            print("-input_file: (String) Path to the file containing the pair to register.")
            print("-model_path: (String) Path to the registration model.\n")
            print("-metric_path: (String) Path to the file for the output metrics. (JSON file).")
            print("KeyWords:\n")
            print("-s: Show the image rather than writing them")
            print("-max_angle: (Int) Maximum angle of rotation to apply when a single image is given.")
            print("-max_translation: (Int) Maximum translation y and x to apply when a single image is given.")
            print("-h: Show this helper section")
            sys.exit()
        elif opt == "-s":
            show = True
        elif opt == "--input_file":
            input_file = arg
        elif opt == "--output_dir":
            output_dir = arg
        elif opt == "--model_path":
            model_path = arg
        elif opt == "--max_angle":
            max_angle = int(arg)
        elif opt == "--max_translation":
            max_translation = int(arg)
        elif opt == "--metric_path":
            metric_path = arg

    # Process Parameters
    if not show:
        os.mkdir(output_dir)
    with open(input_file,"r") as file:
        data_paths = json.load(file)
    model = keras.models.load_model(model_path)
    metrics = dict()

    # Perform Registration
    for idx, paths in data_paths.items():
        paths = [paths["static"]]+[paths["moving"]]
        if paths[1] == "":
            paths = paths[:1]
        moved, moving, static,  ncc = register(model, paths, max_angle, max_translation, target_shape, vessel, seg_model, show=show)
        if not show:
            file_name = output_dir+"/"+str(idx)+".jpg"
            moved = moved[0,:,:,:]*255
            moved = moved.numpy().astype(np.uint8)
            moving = moving[0,:,:,:]*255
            moving = moving.astype(np.uint8)
            static = static[0,:,:,:]*255
            static = static.astype(np.uint8)
            cv2.imwrite(file_name,np.concatenate([moved,static,moving],axis=1))
            metrics[idx] = str(ncc)
    with open(metric_path,"w") as out:
        json.dump(metrics,out)

def compute_metric(moved,static):
    eps = tf.constant(1e-9, 'float32')

    static_mean = tf.reduce_mean(static, axis=[0, 1], keepdims=True)
    moved_mean = tf.reduce_mean(moved, axis=[0, 1], keepdims=True)
    # shape (1, 1, C)

    static_std = tf.math.reduce_std(static, axis=[0, 1], keepdims=True)
    moved_std = tf.math.reduce_std(moved, axis=[0, 1], keepdims=True)
    # shape (1, 1, C)

    static_hat = (static - static_mean)/(static_std + eps)
    moved_hat = (moved - moved_mean)/(moved_std + eps)
    # shape (N, H, W, C)

    ncc = tf.reduce_mean(static_hat * moved_hat)  # shape ()
    ncc = ncc.numpy()
    return ncc

def pre_process(data_paths, target_shape, vessel, max_angle, max_translation, seg_model):
    masks = []
    originals = []
    for path in data_paths:
        original = cv2.imread(path)
        original = reshape_image(original,(256,256))
        original = (original/255).astype(np.float32)
        originals.append(original)
        mask = create_mask(path, target_shape, vessel, model=seg_model)
        mask = cv2.resize(mask, (256,256))
        if len(mask.shape) > 2:
            mask = rgb2gray(mask)
        mask = (mask / 255).astype(np.float32)
        masks.append(mask)
    if len(masks) == 1:
        new_original, angle, tx , ty = augment_data(originals[0], max_angle, max_translation)
        originals.append(new_original)
        new_mask, angle, tx , ty = augment_data(masks[0],max_angle,max_translation)
        masks.append(new_mask)
    return masks, originals

def register(model, data_paths, max_angle, max_translation, target_shape, vessel, seg_model, show=False):
    masks, originals = pre_process(data_paths,target_shape,vessel,max_angle,max_translation,seg_model)
    static = masks[0]
    moving = masks[1]
    # reshape to fit model input shape
    static = static[np.newaxis,:,:,np.newaxis]
    moving = moving[np.newaxis,:,:,np.newaxis]
    transform_param = model({"moving":moving, "static":static},training=False)
    M = build_transformation_matrix(transform_param)
    static = cv2.resize(originals[0],(256,256))
    moving = cv2.resize(originals[1],(256,256))
    static = static[np.newaxis,:,:,:]
    moving = moving[np.newaxis,:,:,:]
    moved = apply_transformation_matrix(M, moving)
    ncc = compute_metric(moved,static)
    if show:
        plot_images(moved, moving, static)
    return moved, moving, static, ncc

if __name__ == "__main__":
    main(sys.argv[1:])

