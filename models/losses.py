from keras import backend as K

def weighted_categorical_crossentropy_loss(weights): 
    
    weights = K.variable(weights)
    def loss(y_true, y_pred):
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    return loss

def jaccard_loss(y_true, y_pred, smooth = 100):
    
    intersection = K.sum(y_true * y_pred, axis=-1)
    union = K.sum(y_true + y_pred, axis=-1) - intersection
    jaccard = (intersection + smooth) / (union + smooth)
    return (1 - jaccard) * smooth

def dice_loss(y_true, y_pred, smooth = 1):
    
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    sum_denom = K.sum(y_true_f) + K.sum(y_pred_f)
    return -2. * (intersection + smooth) / (sum_denom + smooth)