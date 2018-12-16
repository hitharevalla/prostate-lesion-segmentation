from keras.layers import *
from keras.models import *

def unet_padded(
    n_classes, input_width, input_height, 
    path_vgg_weights = None, train_vgg = False,
    data_format = 'channels_first'
):
    
    if data_format == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    img_input = Input(shape = (3, input_height, input_width))
    
    # Block 1
    x = Conv2D(64, (3, 3), 
        kernel_initializer = 'he_normal',
        activation = 'relu', 
        padding = 'same', 
        data_format = data_format, 
        name = 'block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), 
        kernel_initializer = 'he_normal',
        activation = 'relu', 
        padding = 'same', 
        data_format = data_format, 
        name = 'block1_conv2')(x)
    c1 = x
    
    # Block 2
    x = MaxPooling2D((2, 2),
        strides = (2, 2), 
        data_format = data_format,
        name = 'block2_pool')(x)
    x = Conv2D(128, (3, 3), 
        kernel_initializer = 'he_normal',
        activation = 'relu', 
        padding = 'same', 
        data_format = data_format, 
        name = 'block2_conv1')(x)
    x = Conv2D(128, (3, 3), 
        kernel_initializer = 'he_normal',
        activation = 'relu', 
        padding = 'same', 
        data_format = data_format, 
        name = 'block2_conv2')(x)
    c2 = x

    # Block 3
    x = MaxPooling2D((2, 2),
        strides = (2, 2), 
        data_format = data_format,
        name = 'block3_pool')(x)
    x = Conv2D(256, (3, 3), 
        kernel_initializer = 'he_normal',
        activation = 'relu', 
        padding = 'same', 
        data_format = data_format, 
        name = 'block3_conv1')(x)
    x = Conv2D(256, (3, 3),
        kernel_initializer = 'he_normal', 
        activation = 'relu', 
        padding = 'same', 
        data_format = data_format, 
        name = 'block3_conv2')(x)
    c3 = x

    # Block 4
    x = MaxPooling2D((2, 2),
        strides = (2, 2), 
        data_format = data_format,
        name = 'block4_pool')(x)
    x = Conv2D(512, (3, 3), 
        kernel_initializer = 'he_normal',
        activation = 'relu', 
        padding = 'same', 
        data_format = data_format, 
        name = 'block4_conv1')(x)
    x = Conv2D(512, (3, 3), 
        kernel_initializer = 'he_normal',
        activation = 'relu', 
        padding = 'same', 
        data_format = data_format, 
        name = 'block4_conv2')(x)
    c4 = x

    # Block 5
    x = MaxPooling2D((2, 2),
        strides = (2, 2), 
        data_format = data_format,
        name = 'block5_pool')(x)
    x = Conv2D(1024, (3, 3), 
        kernel_initializer = 'he_normal',
        activation = 'relu', 
        padding = 'same', 
        data_format = data_format, 
        name = 'block5_conv1')(x)
    x = Conv2D(1024, (3, 3), 
        kernel_initializer = 'he_normal',
        activation = 'relu', 
        padding = 'same', 
        data_format = data_format, 
        name = 'block5_conv2')(x)
    
    # Block 1
    o = UpSampling2D((2,2), 
        data_format = data_format,
        name = 'd_block1_up')(x)
    o = Conv2D(512, (2, 2),
        kernel_initializer = 'he_normal',
        activation = 'relu', 
        padding = 'same', 
        data_format = data_format,
        name = 'd_block1_conv1')(o)
    o = concatenate([o, c4],
        axis = channel_axis,
        name = 'd_block1_concat')
    o = Conv2D(512, (3, 3),
        kernel_initializer = 'he_normal',
        activation = 'relu', 
        padding = 'same', 
        data_format = data_format,
        name = 'd_block1_conv2')(o)
    o = Conv2D(512, (3, 3),
        kernel_initializer = 'he_normal',
        activation = 'relu', 
        padding = 'same', 
        data_format = data_format,
        name = 'd_block1_conv3')(o)

    # Block 2
    o = UpSampling2D((2,2), 
        data_format = data_format,
        name = 'd_block2_up')(o)
    o = Conv2D(256, (2, 2),
        kernel_initializer = 'he_normal',
        activation = 'relu', 
        padding = 'same', 
        data_format = data_format,
        name = 'd_block2_conv1')(o)
    o = concatenate([o, c3],
        axis = channel_axis,
        name = 'd_block2_concat')
    o = Conv2D(256, (3, 3),
        kernel_initializer = 'he_normal',
        activation = 'relu', 
        padding = 'same', 
        data_format = data_format,
        name = 'd_block2_conv2')(o)
    o = Conv2D(256, (3, 3),
        kernel_initializer = 'he_normal',
        activation = 'relu', 
        padding = 'same',
        data_format = data_format,
        name = 'd_block2_conv3')(o)

    # Block 3
    o = UpSampling2D((2,2), 
        data_format = data_format,
        name = 'd_block3_up')(o)
    o = Conv2D(128, (2, 2),
        kernel_initializer = 'he_normal',
        activation = 'relu', 
        padding = 'same', 
        data_format = data_format,
        name = 'd_block3_conv1')(o)
    o = concatenate([o, c2],
        axis = channel_axis,
        name = 'd_block3_concat')
    o = Conv2D(128, (3, 3),
        kernel_initializer = 'he_normal',
        activation = 'relu', 
        padding = 'same', 
        data_format = data_format,
        name = 'd_block3_conv2')(o)
    o = Conv2D(128, (3, 3),
        kernel_initializer = 'he_normal',
        activation = 'relu', 
        padding = 'same', 
        data_format = data_format,
        name = 'd_block3_conv3')(o)
    
    # Block 4
    o = UpSampling2D((2,2), 
        data_format = data_format,
        name = 'd_block4_up')(o)
    o = Conv2D(64, (2, 2),
        kernel_initializer = 'he_normal',
        activation = 'relu', 
        padding = 'same', 
        data_format = data_format,
        name = 'd_block4_conv1')(o)
    o = concatenate([o, c1],
        axis = channel_axis,
        name = 'd_block4_concat')
    o = Conv2D(64, (3, 3),
        kernel_initializer = 'he_normal',
        activation = 'relu', 
        padding = 'same', 
        data_format = data_format,
        name = 'd_block4_conv2')(o)
    o = Conv2D(64, (3, 3),
        kernel_initializer = 'he_normal',
        activation = 'relu', 
        padding = 'same', 
        data_format = data_format,
        name = 'd_block4_conv3')(o)

    o = Conv2D(n_classes, (1, 1),
        kernel_initializer = 'he_normal',
        padding = 'same', 
        data_format = data_format,
        name = 'd_out_conv')(o)
    
    o_shape = Model(img_input, o).output_shape
    output_height = o_shape[2]
    output_width = o_shape[3]

    o = Reshape((n_classes, output_height * output_width))(o)
    o = Permute((2, 1))(o)
    
    model = Model(img_input, o)
    model.output_width = output_width
    model.output_height = output_height

    return model