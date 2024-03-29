# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 23:55:53 2019

@author: valdis
"""
import decays
from re import findall
import tensorflow as tf, glob
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import LearningRateScheduler
import matplotlib.pyplot as plt
from numpy import around

EPOCHS=60; BASE_RATE=1e-3
CROP_HEIGHT=200; CROP_WIDTH=300; CHANNELS=3
BATCH_SIZE=8; BUFFER_SIZE=4

def plot_rate(history):

    fig_name = history['type'] + '_' + str(around(history['factor'],2))
    
    plt.subplot(211)
    plt.plot(range(1,EPOCHS+1), history['val_accuracy'], label='validation')
    plt.plot(range(1,EPOCHS+1), history['accuracy'], label='training')
    plt.legend(loc=0); plt.xlim([1,EPOCHS])
    plt.xlabel('epochs'); plt.ylabel('accuracy')
    plt.title(fig_name)
        
    plt.subplot(212)
    plt.plot(range(1,EPOCHS+1), history['lr'], label='learning rate')
    plt.legend(loc=0); plt.xlim([1,EPOCHS])
    plt.xlabel('epochs'); plt.ylabel('learning rate')

    plt.subplots_adjust(top=1.5)
    plt.savefig(history['type']+'.png', bbox_inches = "tight", dpi=300)
    
    
def _parse_features(example_item):
    features = {'image': tf.io.FixedLenFeature([], tf.string),
                'label': tf.io.FixedLenFeature([], tf.string)}
    example_item = tf.io.parse_single_example(example_item, features)
    image = tf.image.decode_jpeg(example_item['image'], channels=CHANNELS)/255
    label = tf.image.decode_jpeg(example_item['label'], channels=CHANNELS)/255           
    return image, label

def _crop_image(image, label):   
    image_tensor = tf.stack([image, label], axis=0)
    combined_images = tf.image.random_crop(image_tensor, size = [2, CROP_HEIGHT, CROP_WIDTH, CHANNELS])
    return tf.unstack(combined_images, num=2, axis=0)    
    
def _prepare_datasets(file_list):
    image_dataset = tf.data.TFRecordDataset(file_list)
    return image_dataset.map(_parse_features).map(_crop_image)\
                        .batch(batch_size=BATCH_SIZE).cache()\
                        .shuffle(buffer_size=BUFFER_SIZE).repeat()\
                        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

def create_model(optimizer):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=64, kernel_size=7, padding='same', activation=tf.keras.activations.relu),
        tf.keras.layers.Conv2D(filters=32, kernel_size=1, activation=tf.keras.activations.relu),
        tf.keras.layers.Conv2D(filters=CHANNELS, kernel_size=5, padding = 'same')])
    model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])
    return model
 
def record_number(input_list):
    total_sum = 0
    for string in input_list: 
        digit_list = findall(r'\d+', string)
        item_sum = sum([int(item) for item in digit_list])
        total_sum += item_sum
    return total_sum    

if __name__ == "__main__":   
       
    record_folder = 'F:/SR_folder/record_folder'
    train_list = glob.glob(record_folder + '/train_*.tfrecord')
    validation_list = glob.glob(record_folder + '/validation_*.tfrecord')
    
    train_data = _prepare_datasets(train_list)
    validation_data = _prepare_datasets(validation_list)
    
    validation_number = record_number(validation_list)
    train_number = record_number(train_list)
    
    #%% Decay functions
    
    factor = 0.1; step_drop = 5; border_epoch = 5
    optimizer = SGD(lr=BASE_RATE, decay = 0.0, momentum = 0.0, nesterov = False)
    model = create_model(optimizer)
    
    print('Factor value =', factor)   
    exp_decay = decays.ExponentDecay(learning_rate=BASE_RATE, exp_power=factor)
    exp_decay_border = decays.ExponentDecay(border_epoch=border_epoch, learning_rate=BASE_RATE, exp_power=factor)
	 
    step_decay = decays.StepDecay(learning_rate=BASE_RATE, step_factor=factor, step_drop=step_drop)
    step_decay_border = decays.StepDecay(border_epoch=border_epoch, learning_rate=BASE_RATE, step_factor=factor, step_drop=step_drop)
   
    callbacks = LearningRateScheduler(step_decay_border)
    history = model.fit(train_data, epochs=EPOCHS, verbose=1, validation_data=validation_data, callbacks=[callbacks],
                           steps_per_epoch=int(train_number/float(BATCH_SIZE)), validation_steps=int(validation_number/float(BATCH_SIZE)))
            
    history_dict = history.history
    history_dict.update({'factor' : around(factor,3), 'type': 'step_decay_border'})
    plot_rate(history_dict)
      
#    Epoch 1/60
#    379/379 [==============================] - 23s 62ms/step - loss: 0.2362 - accuracy: 0.2092 - val_loss: 0.1796 - val_accuracy: 0.2271
#    Epoch 2/60
#    379/379 [==============================] - 23s 61ms/step - loss: 0.0700 - accuracy: 0.2538 - val_loss: 0.0174 - val_accuracy: 0.3007
#    Epoch 3/60
#    379/379 [==============================] - 23s 61ms/step - loss: 0.0120 - accuracy: 0.5069 - val_loss: 0.0109 - val_accuracy: 0.5760
#    Epoch 4/60
#    379/379 [==============================] - 23s 61ms/step - loss: 0.0102 - accuracy: 0.6425 - val_loss: 0.0101 - val_accuracy: 0.6230
#    Epoch 5/60
#    379/379 [==============================] - 23s 61ms/step - loss: 0.0096 - accuracy: 0.6725 - val_loss: 0.0096 - val_accuracy: 0.6350
#    ...    
#    Epoch 55/60
#    379/379 [==============================] - 23s 62ms/step - loss: 0.0081 - accuracy: 0.7312 - val_loss: 0.0083 - val_accuracy: 0.6867
#    Epoch 56/60
#    379/379 [==============================] - 23s 62ms/step - loss: 0.0081 - accuracy: 0.7312 - val_loss: 0.0083 - val_accuracy: 0.6867
#    Epoch 57/60
#    379/379 [==============================] - 23s 62ms/step - loss: 0.0081 - accuracy: 0.7312 - val_loss: 0.0083 - val_accuracy: 0.6867
#    Epoch 58/60
#    379/379 [==============================] - 23s 62ms/step - loss: 0.0081 - accuracy: 0.7312 - val_loss: 0.0083 - val_accuracy: 0.6867
#    Epoch 59/60
#    379/379 [==============================] - 23s 62ms/step - loss: 0.0081 - accuracy: 0.7312 - val_loss: 0.0083 - val_accuracy: 0.6867
#    Epoch 60/60
#    379/379 [==============================] - 23s 62ms/step - loss: 0.0081 - accuracy: 0.7312 - val_loss: 0.0083 - val_accuracy: 0.6867
