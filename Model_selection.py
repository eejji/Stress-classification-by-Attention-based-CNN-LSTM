#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from keras.utils import to_categorical
from sklearn.model_selection import KFold
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import clone_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, LSTM, Dense, Flatten, Conv1D,TimeDistributed, Reshape
from tensorflow.keras.layers import MaxPooling1D, BatchNormalization, Dropout,Activation
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,roc_auc_score
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# set early stopping 
def set_early_stopping():
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, verbose=1, min_delta=0.005, restore_best_weights=True)
    
    return early_stopping

def train_val_test_split(data, label, stratify=False):
    
    if stratify == False:
        x_trainval, x_test, y_trainval, y_test = train_test_split(data, label, test_size=0.2, random_state=128)
        x_train, x_val, y_train, y_val = train_test_split(x_trainval, y_trainval, test_size=0.25,  random_state=128)
    else :  
        x_trainval, x_test, y_trainval, y_test = train_test_split(data, label, test_size=0.2, random_state=128, stratify=label)
        x_train, x_val, y_train, y_val = train_test_split(x_trainval, y_trainval, test_size=0.25,  random_state=128, stratify=y_trainval)
    
    
    return x_train, y_train, x_val, y_val, x_test, y_test




class DotProductAttention(Layer):
    def __init__(self, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)

    def call(self, query, key, value):
        # Compute the dot product attention scores
        scores = tf.matmul(query, key, transpose_b=True)
        
        # Scale scores
        depth = tf.cast(tf.shape(key)[-1], tf.float32)
        scores = scores / tf.math.sqrt(depth)
        
        # Softmax to obtain attention weights
        weights = tf.nn.softmax(scores, axis=-1)
        
        # Multiply the weights with the values
        attention_output = tf.matmul(weights, value)
        
        return attention_output

def fusion_model(beat_shape, rhythm_shape):
    
    input_beat = Input(shape=(beat_shape,1), name = 'input_beat')
    input_rhythm = Input(shape=(rhythm_shape,1), name = 'input_rhythm')
    
    conv1_B = Conv1D(16, 11, activation=None, padding='same')(input_beat)
    conv2_B = Conv1D(16, 11, activation=None, padding='same')(conv1_B)
    bn1_B = BatchNormalization()(conv2_B)
    act_B = Activation('relu')(bn1_B)
    residual_B = add([conv1_B, act_B])
    max1_B = MaxPooling1D(2,2)(residual_B)
    
    conv1_B_1 = Conv1D(32, 9, activation=None, padding='same')(max1_B)
    conv2_B_1 = Conv1D(32, 9, activation=None, padding='same')(conv1_B_1)
    bn1_B_1 = BatchNormalization()(conv2_B_1)
    act_B_1 = Activation('relu')(bn1_B_1)
    residual_B_1 = add([conv1_B_1, act_B_1])
    max1_B_1 = MaxPooling1D(2,2)(residual_B_1)
    
    
    conv1_B_2 = Conv1D(64, 7, activation=None, padding='same')(max1_B_1)
    conv2_B_2 = Conv1D(64, 7, activation=None, padding='same')(conv1_B_2)
    bn1_B_2 = BatchNormalization()(conv2_B_2)
    act_B_2 = Activation('relu')(bn1_B_2)
    residual_B_2 = add([conv1_B_2, act_B_2])
    max1_B_2 = MaxPooling1D(2,2)(residual_B_2)
    
    
    conv1_B_3 = Conv1D(128, 5, activation=None, padding='same')(max1_B_2)
    conv2_B_3 = Conv1D(128, 5, activation=None, padding='same')(conv1_B_3)
    bn1_B_3 = BatchNormalization()(conv2_B_3)
    act_B_3 = Activation('relu')(bn1_B_3)
    residual_B_3 = add([conv1_B_3, act_B_3])
    max1_B_3 = MaxPooling1D(2,2)(residual_B_3)
    
    
    conv1_B_4 = Conv1D(256, 3, activation=None, padding='same')(max1_B_3)
    conv2_B_4 = Conv1D(256, 3, activation=None, padding='same')(conv1_B_4)
    bn1_B_4 = BatchNormalization()(conv2_B_4)
    act_B_4 = Activation('relu')(bn1_B_4)
    residual_B_4 = add([conv1_B_4, act_B_4])
    max1_B_4 = MaxPooling1D(2,2)(residual_B_4)

    conv1_B_5 = Conv1D(128, 1, activation=None, padding='same')(max1_B_4)
    bn1_B_5 = BatchNormalization()(conv1_B_5)
    act_B_5 = Activation('relu')(bn1_B_5)
    residual_B_5 = add([conv1_B_5, act_B_5])
    
    re_B = Reshape((1, residual_B_5.shape[1], residual_B_5.shape[2]))(residual_B_5)
    t1_B = TimeDistributed(Flatten())(re_B)
    x_B_LSTM = LSTM(256, return_sequences=True)(t1_B)
#     x_B = Dropout(0.05)(x_B_LSTM)
    Dotproduct_1 = DotProductAttention()(x_B_LSTM, x_B_LSTM,x_B_LSTM)
    x_B_LSTM2 = LSTM(50, return_sequences=True)(Dotproduct_1)  
    Dotproduct_2 = DotProductAttention()(x_B_LSTM2,x_B_LSTM2,x_B_LSTM2)
    
    GAP_B = GlobalAveragePooling1D()(Dotproduct_2)
    
    
    # Rhythm
    
    conv1_R = Conv1D(16, 15, activation=None, padding='same')(input_rhythm)
    conv2_R = Conv1D(16, 15, activation=None, padding='same')(conv1_R)
    bn1_R = BatchNormalization()(conv2_R)
    act_R = Activation('relu')(bn1_R)
    residual_R = add([conv1_R, act_R])
    max1_R = MaxPooling1D(2,2)(residual_R)
    
    conv1_R_1 = Conv1D(32, 13, activation=None, padding='same')(max1_R)
    conv2_R_1 = Conv1D(32, 13, activation=None, padding='same')(conv1_R_1)
    bn1_R_1 = BatchNormalization()(conv2_R_1)
    act_R_1 = Activation('relu')(bn1_R_1)
    residual_R_1 = add([conv1_R_1, act_R_1])
    max1_R_1 = MaxPooling1D(2,2)(residual_R_1)
    
    
    conv1_R_2 = Conv1D(64, 11, activation=None, padding='same')(max1_R_1)
    conv2_R_2 = Conv1D(64, 11, activation=None, padding='same')(conv1_R_2)
    bn1_R_2 = BatchNormalization()(conv2_R_2)
    act_R_2 = Activation('relu')(bn1_R_2)
    residual_R_2 = add([conv1_R_2, act_R_2])
    max1_R_2 = MaxPooling1D(2,2)(residual_R_2)
    
    
    conv1_R_3 = Conv1D(128, 9, activation=None, padding='same')(max1_R_2)
    conv2_R_3 = Conv1D(128, 9, activation=None, padding='same')(conv1_R_3)
    bn1_R_3 = BatchNormalization()(conv2_R_3)
    act_R_3 = Activation('relu')(bn1_R_3)
    residual_R_3 = add([conv1_R_3, act_R_3])
    max1_R_3 = MaxPooling1D(2,2)(residual_R_3)
    
    
    conv1_R_4 = Conv1D(256, 7, activation=None, padding='same')(max1_R_3)
    conv2_R_4 = Conv1D(256, 7, activation=None, padding='same')(conv1_R_4)
    bn1_R_4 = BatchNormalization()(conv2_R_4)
    act_R_4 = Activation('relu')(bn1_R_4)
    residual_R_4 = add([conv1_R_4, act_R_4])
    max1_R_4 = MaxPooling1D(2,2)(residual_R_4)
    
    conv1_R_5 = Conv1D(512, 5, activation=None, padding='same')(max1_R_4)
    conv2_R_5 = Conv1D(512, 5, activation=None, padding='same')(conv1_R_5)
    bn1_R_5 = BatchNormalization()(conv2_R_5)
    act_R_5 = Activation('relu')(bn1_R_5)
    residual_R_5 = add([conv1_R_5, act_R_5])
    max1_R_5 = MaxPooling1D(2,2)(residual_R_5)
    
    conv1_R_6 = Conv1D(256, 3, activation=None, padding='same')(max1_R_5)
    conv2_R_6 = Conv1D(256, 3, activation=None, padding='same')(conv1_R_6)
    bn1_R_6 = BatchNormalization()(conv2_R_6)
    act_R_6 = Activation('relu')(bn1_R_6)
    residual_R_6 = add([conv1_R_6, act_R_6])
    max1_R_6 = MaxPooling1D(2,2)(residual_R_6)
    
    
    conv1 = Conv1D(128, 1, activation=None, padding='same')(max1_R_6)
    bn1 = BatchNormalization()(conv1)
    act = Activation('relu')(bn1)
    residual = add([conv1, act])
    
    re_R = Reshape((1, residual.shape[1], residual.shape[2]))(residual)
    t1_R = TimeDistributed(Flatten())(re_R)
    x_R_LSTM = LSTM(256, return_sequences=True)(t1_R)
#     x_R = Dropout(0.05)(x_R_LSTM)
    Dotporduct_R = DotProductAttention()(x_R_LSTM,x_R_LSTM,x_R_LSTM)
    x_R_LSTM2 = LSTM(50, return_sequences=True)(Dotporduct_R)  
    Dotporduct_R2 = DotProductAttention()(x_R_LSTM2,x_R_LSTM2,x_R_LSTM2)
    
    GAP_R = GlobalAveragePooling1D()(Dotporduct_R2)
    
    #fusion
    concatenated = Concatenate()([GAP_B, GAP_R])
    den1 = Dense(128, activation='relu')(concatenated)
    drop = Dropout(0.05)(den1)
    den2 = Dense(64, activation='relu')(drop)
    
    fusion_output = Dense(3, activation='softmax')(den2)
    
    fusion_model = Model(inputs=[input_beat, input_rhythm], outputs= fusion_output)
    fusion_model.summary()
    
    return fusion_model
    