from keras import backend as K

def iou(y_true, y_pred):
    
    y_true = K.cast(K.flatten(K.argmax(y_true)), 'float32')
    y_pred = K.cast(K.flatten(K.argmax(y_pred)), 'float32')  
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    return intersection / union

def dice(y_true, y_pred):
    
    y_true = K.cast(K.flatten(K.argmax(y_true)), 'float32')
    y_pred = K.cast(K.flatten(K.argmax(y_pred)), 'float32')
    num = 2 * K.sum(y_true * y_pred)
    denom = K.sum(y_true) + K.sum(y_pred)
    return num / denom