from keras.layers import *
from keras.models import *

def vgg_segnet(
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
        trainable = train_vgg,
        name = 'block1_conv1')(img_input)
    x = Conv2D(64, (3, 3),
        kernel_initializer = 'he_normal',
        activation = 'relu', 
        padding = 'same', 
        data_format = data_format, 
        trainable = train_vgg,
        name = 'block1_conv2')(x)
    x = MaxPooling2D((2, 2),
        strides = (2, 2), 
        data_format = data_format,
        name = 'block1_pool')(x)
    
    # Block 2
    x = Conv2D(128, (3, 3), 
        kernel_initializer = 'he_normal',
        activation = 'relu', 
        padding = 'same', 
        data_format = data_format, 
        trainable = train_vgg,
        name = 'block2_conv1')(x)
    x = Conv2D(128, (3, 3), 
        kernel_initializer = 'he_normal',
        activation = 'relu', 
        padding = 'same', 
        data_format = data_format, 
        trainable = train_vgg,
        name = 'block2_conv2')(x)
    x = MaxPooling2D((2, 2),
        strides = (2, 2), 
        data_format = data_format,
        name = 'block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), 
        kernel_initializer = 'he_normal',
        activation = 'relu', 
        padding = 'same', 
        data_format = data_format, 
        trainable = train_vgg,
        name = 'block3_conv1')(x)
    x = Conv2D(256, (3, 3), 
        kernel_initializer = 'he_normal',
        activation = 'relu', 
        padding = 'same', 
        data_format = data_format, 
        trainable = train_vgg,
        name = 'block3_conv2')(x)
    x = Conv2D(256, (3, 3), 
        kernel_initializer = 'he_normal',
        activation = 'relu', 
        padding = 'same', 
        data_format = data_format, 
        trainable = train_vgg,
        name = 'block3_conv3')(x)
    x = MaxPooling2D((2, 2),
        strides = (2, 2), 
        data_format = data_format,
        name = 'block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), 
        kernel_initializer = 'he_normal',
        activation = 'relu', 
        padding = 'same', 
        data_format = data_format, 
        trainable = train_vgg,
        name = 'block4_conv1')(x)
    x = Conv2D(512, (3, 3), 
        kernel_initializer = 'he_normal',
        activation = 'relu', 
        padding = 'same', 
        data_format = data_format, 
        trainable = train_vgg,
        name = 'block4_conv2')(x)
    x = Conv2D(512, (3, 3), 
        kernel_initializer = 'he_normal',
        activation = 'relu', 
        padding = 'same', 
        data_format = data_format, 
        trainable = train_vgg,
        name = 'block4_conv3')(x)
    x = MaxPooling2D((2, 2),
        strides = (2, 2), 
        data_format = data_format,
        name = 'block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), 
        kernel_initializer = 'he_normal',
        activation = 'relu', 
        padding = 'same', 
        data_format = data_format, 
        trainable = train_vgg,
        name = 'block5_conv1')(x)
    x = Conv2D(512, (3, 3), 
        kernel_initializer = 'he_normal',
        activation = 'relu', 
        padding = 'same', 
        data_format = data_format, 
        trainable = train_vgg,
        name = 'block5_conv2')(x)
    x = Conv2D(512, (3, 3), 
        kernel_initializer = 'he_normal',
        activation = 'relu', 
        padding = 'same', 
        data_format = data_format, 
        trainable = train_vgg,
        name = 'block5_conv3')(x)
    x = MaxPooling2D((2, 2),
        strides = (2, 2), 
        data_format = data_format,
        name = 'block5_pool')(x)
    
    vgg = Model(img_input, x)
    if path_vgg_weights:
        vgg.load_weights(path_vgg_weights, by_name = True)

    # Block 1
    o = UpSampling2D((2,2), 
        data_format = data_format,
        name = 'd_block1_up')(x)
    o = Conv2D(512, (3, 3), 
        kernel_initializer = 'he_normal',
        padding = 'same', 
        data_format = data_format,
        name = 'd_block1_conv1')(o)
    o = BatchNormalization(
        axis = channel_axis,
        name = 'd_block1_batchnorm1')(o)
    o = Activation('relu',
        name = 'd_block1_relu1')(o)
    o = Conv2D(512, (3, 3), 
        kernel_initializer = 'he_normal',
        padding = 'same', 
        data_format = data_format,
        name = 'd_block1_conv2')(o)
    o = BatchNormalization(
        axis = channel_axis,
        name = 'd_block1_batchnorm2')(o)
    o = Activation('relu',
        name = 'd_block1_relu2')(o)
    o = Conv2D(512, (3, 3), 
        kernel_initializer = 'he_normal',
        padding = 'same', 
        data_format = data_format,
        name = 'd_block1_conv3')(o)
    o = BatchNormalization(
        axis = channel_axis,
        name = 'd_block1_batchnorm3')(o)
    o = Activation('relu',
        name = 'd_block1_relu3')(o)

    # Block 2
    o = UpSampling2D((2,2),
        data_format = data_format,
        name = 'd_block2_up')(o)
    o = Conv2D(512, (3, 3), 
        kernel_initializer = 'he_normal',
        padding = 'same', 
        data_format = data_format,
        name = 'd_block2_conv1')(o)
    o = BatchNormalization(
        axis = channel_axis,
        name = 'd_block2_batchnorm1')(o)
    o = Activation('relu',
        name = 'd_block2_relu1')(o)
    o = Conv2D(512, (3, 3), 
        kernel_initializer = 'he_normal',
        padding = 'same', 
        data_format = data_format,
        name = 'd_block2_conv2')(o)
    o = BatchNormalization(
        axis = channel_axis,
        name = 'd_block2_batchnorm2')(o)
    o = Activation('relu',
        name = 'd_block2_relu2')(o)
    o = Conv2D(512, (3, 3), 
        kernel_initializer = 'he_normal',
        padding = 'same', 
        data_format = data_format,
        name = 'd_block2_conv3')(o)
    o = BatchNormalization(
        axis = channel_axis,
        name = 'd_block2_batchnorm3')(o)
    o = Activation('relu',
        name = 'd_block2_relu3')(o)

    # Block 3
    o = UpSampling2D((2,2),
        data_format = data_format,
        name = 'd_block3_up')(o)
    o = Conv2D(256, (3, 3), 
        kernel_initializer = 'he_normal',
        padding = 'same', 
        data_format = data_format,
        name = 'd_block3_conv1')(o)
    o = BatchNormalization(
        axis = channel_axis,
        name = 'd_block3_batchnorm1')(o)
    o = Activation('relu',
        name = 'd_block3_relu1')(o)
    o = Conv2D(256, (3, 3), 
        kernel_initializer = 'he_normal',
        padding = 'same', 
        data_format = data_format,
        name = 'd_block3_conv2')(o)
    o = BatchNormalization(
        axis = channel_axis,
        name = 'd_block3_batchnorm2')(o)
    o = Activation('relu',
        name = 'd_block3_relu2')(o)
    o = Conv2D(256, (3, 3), 
        kernel_initializer = 'he_normal',
        padding = 'same', 
        data_format = data_format,
        name = 'd_block3_conv3')(o)
    o = BatchNormalization(
        axis = channel_axis,
        name = 'd_block3_batchnorm3')(o)
    o = Activation('relu',
        name = 'd_block3_relu3')(o)

    # Block 4
    o = UpSampling2D((2,2),
        data_format = data_format,
        name = 'd_block4_up')(o)
    o = Conv2D(128, (3, 3), 
        kernel_initializer = 'he_normal',
        padding = 'same', 
        data_format = data_format,
        name = 'd_block4_conv1')(o)
    o = BatchNormalization(
        axis = channel_axis,
        name = 'd_block4_batchnorm1')(o)
    o = Activation('relu',
        name = 'd_block4_relu1')(o)
    o = Conv2D(128, (3, 3), 
        kernel_initializer = 'he_normal',
        padding = 'same', 
        data_format = data_format,
        name = 'd_block4_conv2')(o)
    o = BatchNormalization(
        axis = channel_axis,
        name = 'd_block4_batchnorm2')(o)
    o = Activation('relu',
        name = 'd_block4_relu2')(o)

    # Block 5
    o = UpSampling2D((2,2),
        data_format = data_format,
        name = 'd_block5_up')(o)
    o = Conv2D(64, (3, 3), 
        kernel_initializer = 'he_normal',
        padding = 'same', 
        data_format = data_format,
        name = 'd_block5_conv1')(o)
    o = BatchNormalization(
        axis = channel_axis,
        name = 'd_block5_batchnorm1')(o)
    o = Activation('relu',
        name = 'd_block5_relu1')(o)
    o = Conv2D(n_classes, (1, 1), 
        kernel_initializer = 'he_normal',
        padding = 'valid', 
        data_format = data_format,
        name = 'd_block5_conv2')(o)
    o = BatchNormalization(
        axis = channel_axis,
        name = 'd_block5_batchnorm2')(o)
    
    o_shape = Model(img_input, o).output_shape
    output_height = o_shape[2]
    output_width = o_shape[3]

    o = Reshape((n_classes, output_height * output_width))(o)
    o = Permute((2, 1))(o)
    o = Activation('softmax',
        name = 'd_softmax')(o)
    
    model = Model(img_input, o)
    model.output_width = output_width
    model.output_height = output_height

    return model