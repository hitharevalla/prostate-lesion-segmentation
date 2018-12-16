from keras.models import *
from keras.utils import plot_model
import matplotlib.pyplot as plt

from load_data import *
from model_definitions.segnet import *
from model_definitions.bayesian_segnet import *
from model_definitions.vgg_segnet import *
from model_definitions.vgg_bayesian_segnet import *
from model_definitions.unet import *
from model_definitions.unet_padded import *
from model_definitions.vgg_unet import *
from model_definitions.vgg_unet_padded import *
from utils import *

def train(path_train_images, path_train_annotations, train_batch_size, 
    path_val_images, path_val_annotations, val_batch_size,
    n_classes, input_width, input_height, 
    model_name, epochs, loss, optimizer, metrics, 
    path_save_weights, path_vgg_weights = None, train_vgg = False, 
    path_load_weights = None, resume_epoch = 1):
    
    initialize_dir(path_save_weights) # If directory exists, will delete current files

    model_architectures = {
        'segnet': segnet,
        'bayesian_segnet': bayesian_segnet,
        'vgg_segnet': vgg_segnet,
        'vgg_bayesian_segnet': vgg_bayesian_segnet,
        'unet': unet,
        'unet_padded': unet_padded,
        'vgg_unet': vgg_unet,
        'vgg_unet_padded': vgg_unet_padded,
    }
    model_func = model_architectures[model_name]
     
    if path_vgg_weights:
        m = model_func(n_classes, input_width, input_height, path_vgg_weights = path_vgg_weights, train_vgg = train_vgg)
    else:
        m = model_func(n_classes, input_width, input_height)
        
    plot_model(m, show_shapes = True, to_file = 'model.png')
    
    m.compile(loss = loss, optimizer = optimizer, metrics = metrics)

    if path_load_weights:
        m.load_weights(path_load_weights)

    print('Model output shape', m.output_shape)

    output_height = m.output_height
    output_width = m.output_width

    train_generator = DataGenerator(
        path_train_images, path_train_annotations, 
        train_batch_size, n_classes, 
        input_height, input_width, output_height, output_width
    )
    val_generator = DataGenerator(
        path_val_images, path_val_annotations, 
        val_batch_size, n_classes, 
        input_height, input_width, output_height, output_width
    )
    
    X_train = train_generator.X
    y_train = train_generator.y
    X_val = val_generator.X
    y_val = val_generator.y

    train_loss = []
    val_loss = []
    train_dice = []
    val_dice = []
    
    for e in range(resume_epoch - 1, epochs):
        print('Epoch', e)
        e_history = m.fit_generator(
            train_generator, epochs = 1, verbose = 1,
            validation_data = val_generator, validation_steps = 1,
        )
                
        train_loss.append(e_history.history['loss'])
        val_loss.append(e_history.history['val_loss'])
        train_dice.append(e_history.history['dice'])
        val_dice.append(e_history.history['val_dice'])
        
        m.save_weights(path_save_weights + '.' + str(e))
        
    fig = plt.figure()
    plt.plot(train_loss, 'k-', label = "Train")
    plt.plot(val_loss, 'k--', label = "Validation")
    plt.legend(loc = 'upper left')
    plt.title("Loss vs. Epochs", fontsize = 16, fontweight = 'bold')
    plt.xlabel("Epochs")
    plt.ylabel("Weighted Categorical Crossentropy Loss")
    plt.show()
    fig.savefig('loss.' + str(epochs) + 'e.png', dpi = 300)
    
    fig = plt.figure()
    plt.plot(train_dice, 'k-', label = "Train")
    plt.plot(val_dice, 'k--', label = "Validation")
    plt.legend(loc = 'upper left')
    plt.title("Dice Score vs. Epochs", fontsize = 16, fontweight = 'bold')
    plt.xlabel("Epochs")
    plt.ylabel("Dice Score")
    plt.show()
    fig.savefig('dice.' + str(epochs) + 'e.png', dpi = 300)