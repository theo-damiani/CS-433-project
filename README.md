# Unsupervised Non-Deformable Retina Images Registration Using Neural Network

### Authors: *Jérémy Baffou, Noah Kaltenrieder and Théo Damiani*

Abstract-This project has been part of the Machine Learning course at EPFL by Nicolas Flammarion and Martin Jaggi. It will focus on the broad topic of image registration. A solution will be developed using an unsupervised neural network on pairs of retina images. Allowing only rigid transformations for the main part, the network will calculate the distance between a moving image and another considered as fixed on which we want to align. The transformation matrix found by the network will be then applied to the moving image. Finally, an index will indicate the level of confidence to quantify the similarity between these two images and to know if they come from the same person.

##

The project uses real health data of patients across Switzerland. Therefore for privacy reasons, our model's weights will not be present in the repository. But we can offer you the alternate model that we trained on the public dataset STARE project. We also provide public retina photography so you can test our model with it.

##

The code is ready-to-use, you only need to clone the repo and try the script as follows:
Refer to the help section below if needed.

### *On the SIB cluster with real data:*

Preprocessing is needed:
```
python data_processing.py --input_dir=... 
```

Train the model with:
```
python script.py --params=...
```

See result:
```
python register.py --params=...
```


### *On local machine with STARE dataset:*
As the STARE project regroups public data, the model has been saved and pushed in the repository. 
So preprocessing and training step is not needed.

See result:
```
python register.py --params=...
```
##

## Help section:

### registry.py:
In the input file to the registration script (called registration_input.json here), if you want to see artificial augmentation and registration just set static field to the path to the image to register and the other one (moving) to "". If you want to register two images between them, set static and moving to the corresponding paths.
Provided an input file, this script will make the rigid registration of all image pairs provided. It will also output a file with the metrics (ncc) between the registered image and the static one. Check the github to see an example of input file. The order of the images is moved, static, moving.\
            **Parameters:**\
            -input_file: (String) Path to the file containing the pair to register.\
            -model_path: (String) Path to the registration model.\
            -metric_path: (String) Path to the file for the output metrics. (JSON file).\
            **KeyWords:**\
            -s: Show the image rather than writing them.\
            -max_angle: (Int) Maximum angle of rotation to apply when a single image is given.\
            -max_translation: (Int) Maximum translation y and x to apply when a single image is given.\
            -h: Show this helper section.\

### data_processing.py:
Format and process the data in preparation for model training. This pipeline will process the data in the given folder, merged with the other extra input folders if given, but keeping only shared patients. The first processing is resizing to target shape with black padding to preserve ratio and extract vessel binary masks. Then if needed it will perform data augmentation, i.e. in the cases where there is a single image" for a given type so no registration is possible. It will also perform randomly some data augmentation to avoid bias. A json file will be created with the mapping information between the old data set and the new one.
The format of this file can be looked at in the github. \
**Parameters:**\
-input_dir: (String) Path to the input directory.\
-output_dir: (String) Path to the output directory (will be created).\
-log_dir: (String) Path to the directory for the mapping file.\
**Keywords:**\
-target_shape: (int,int) Target shape for the processed images.\
-vessel: (bool) Indicate if we want to use the segmentation deepnet or not.\
-max_angle: (int) Maximum angle for data augmentation.\
-max_translation: (int) Maximum x and y translation in data augmentation. \
-p_augment: (float) Probability of randomly augmenting valid data.\
-model_path: (String) Path to the model for vessel segmentation, required if vessel is True.\
-extra_input_dirs: (String) Pathes to the other directories if multi folder\
is wanted. Should be in format : path1,path2,...\

### train.py:
Perform model training on the given data. \
**Parameters:**\
-input_dir: (String) Path to the formated data.\
-output_path: (String) Path where the model will be saved should end with the model name.\
**Keywords:**\
-target_shape: (int,int) Shape of the masks if resizing is needed. (default: (512,512)).\
-model_type: (String) Model which will be trained. Can be chosen from:.\
--standard: Simple architecture with Convolutionnal layers and then dense layers. (default).\
--alternative: Similar but alternative architecture to standard.\
--fcn: Fully convolutionnal layers model.\
--extended: Similar to standard but way more deep.\
-balance: (float) Balance between training and test set. (default=0.9).\
-seed: (int) Seed for the numpy random generator. (default=1).\
-epoch: (int) Number of epoch. (default=50).\
-batch_size: (int) Size of the batches of image pair. (default=8).\
-lr: (float) Learning rate. (default=0.9).\
**Output:**\
This pipeline will output the saved model to the given output path.

##

Explanation of the code:
> local_training/ \
> -- Data_Processing.ipynb : *The notebook version of data_processing.py*\
> -- Local_Training.ipynb : *The notebook that trains the model that performs image registration, save the weights obtained and show some result examples*\
> -- Mask_Registration.ipynb : *The notebook that trains the model that performs image registration, based on the mask of the vessels in the eyes*\
> -- Model_Loader.ipynb : *The notebook that import already saved models to reuse them*\
> -- Pipeline.ipynb : *A preliminary notebook that tried image registration with rigid transformation on MNIST dataset and STARE dataset*\
> -- Rigid.ipynb : *The notebook version of rigid_transform.py*\
> -- Vessel Segmentation.ipynb : *The notebook that performs a segmentation of the vessels in the eyes using opencv and returns a mask of them*\
> -- results/model \
> -- -- masks_200_epochs : *The saved model that was trained with masks, based on image processing methods, with 200 epochs*\
> -- -- vessel_200_epochs : *The saved model that was trained with masks, based on UNet results, with 200 epochs*\
> scripts/ \
> -- data_processing.py : *Preprocesses the input images including multiple transformations (reshape, normalization, mask creation, ...) so that we can use them in the Neural Network*\
> -- loss.py : *The different loss functions that we have used*\
> -- models.py : *Multiples functions to create the model that we want to use as a deep learning network for image registration*\
> -- rigid_transform.py : *Multiple functions to define rigid transformation for the Neural Network*\
> -- train.py : *Performs the training of the chosen registration model on the given data and saves the result*\
> -- utils.py : *Different utils functions e.g. show_images, plot_images, split_dataset, ...*

##

The authors gratefully thank our supervisor Mattia Tomasoni.
