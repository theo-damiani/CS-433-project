import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
from scripts.utils import *
from scripts.models import *
from scripts.rigid_transform import *
from scripts.loss import *
import sys
import getopt

def main(argv):
    # Parse arguments
    try:
        opts, args = getopt.getopt(argv,"h",["input_dir=","output_path","target_shape=",
                                 "balance=","seed=", "epoch=", "batch_size=", "model_type=", 
                                 "lr="])
    except getopt.GetoptError:
        print(argv)
        print("Error in inputs, please be sure to provide valid keywords (refer to help section)")
        sys.exit(1)
    input_dir = ""
    output_path = ""
    model_type = "standard"
    target_shape =(512,512)
    balance = 0.9
    seed = 1
    epoch = 50
    batch_size = 8
    lr = 0.1
    for opt, arg in opts:
        if opt == "-h":
            print("\n")
            print("Help section:\n")
            print("Perform model training on the given data.\n") 
            print("Parameters:") 
            print("-input_dir: (String) Path to the formated data.") 
            print("-output_path: (String) Path where the model will be saved should end with the model name.\n")
            print("Keywords:") 
            print("-target_shape: (int,int) Shape of the masks if resizing is needed. (default: (512,512))") 
            print("-model_type: (String) Model which will be trained. Can be chosen from:") 
            print("\t-standard: Simple architecture with Convolutionnal layers and then dense layers. (default)") 
            print("\t-alternative: Similiar but alternative architecture to standard.") 
            print("\t-fcn: Fully convolutionnal layers model.") 
            print("\t-extended: Similar to standard but way more deep.") 
            print("-balance: (float) Balance between training and test set. (default=0.9)") 
            print("-seed: (int) Seed for the numpy random generator. (default=1)")  
            print("-epoch: (int) Number of epoch. (default=50)") 
            print("-batch_size : (int) Size of the batches of image pair. (default=8)") 
            print("-lr: (float) Learning rate. (default=0.9)\n")
            print("Output:") 
            print("This pipeline will output the saved model to the given output path.") 
            sys.exit()
        elif opt == "--input_dir":
            input_dir = arg
        elif opt == "--output_path":
            output_path = arg
        elif opt == "--model_type":
            model_type = arg
        elif opt == "--target_shape":
            width = int(arg.split(",")[0][1:])
            height = int(arg.split(",")[1][:-1])
            target_shape = (width,height)
        elif opt == "--balance":
            balance = float(arg)
        elif opt == "--seed":
            seed = int(arg)
        elif opt == "--epoch":
            epoch = int(arg)
        elif opt == "--batch_size":
            batch_size = int(arg)
        elif opt == "--lr":
            lr = int(arg)
    # Begin training
    data = load_masks(input_dir,target_size=target_shape)
    train, test = split_dataset(data,balance=balance,seed=seed)
    print(f"We have {train.shape[0]} training image pairs and {test.shape[0]} test image pairs.")
    model_post_training = training(train,epoch,batch_size,lr,target_shape,model_type=model_type)
    # Save model
    print(f"Saving model")
    model_post_training.save(output_path)
    print(f"The model has been saved in : {output_path}")

def choose_model(model_type, input_shape):
    """ Output the chosen model architecture. """
    if model_type == "standard":
        return deep_net(input_shape=input_shape)
    elif model_type == "fcn":
        return fc_net(input_shape=input_shape)
    elif model_type == "alternative":
        return simple_cnn(input_shape=input_shape)
    else: 
        return model_extended(input_shape=input_shape)

def training(train_data,epochs,batch_size,lr,size_img,model_type="standard"):
    """ Perfom the training of the choosen registration model on the given data. """
    # Shuffle and batch the dataset.
    from_tensor_slices = tf.data.Dataset.from_tensor_slices
    x_train = from_tensor_slices(train_data).shuffle(10000).batch(batch_size)
    model = choose_model(model_type,size_img+(1,))
    # optimizer set up
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    criterion = ncc_loss
    # Define the metrics to track training and testing losses.
    m_train = tf.keras.metrics.Mean(name='loss_train')
    for epoch in range(epochs):
        m_train.reset_states()
        for i, pair in enumerate(x_train):
            moving = pair[:,0,:,:,:]
            static = pair[:,1,:,:,:]
            loss_train = train_step(model, moving, static, criterion,
                                    optimizer)
            m_train.update_state(loss_train)
        print(f"Epoch: {epoch+1}/{epochs}\tTrain Loss: {m_train.result()}")
    print('\n')
    return model

if __name__ == "__main__":
    main(sys.argv[1:])
