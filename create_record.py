# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 21:44:06 2019

@author: valdis
"""
import os, glob, cv2
import tensorflow as tf

def _bytes_feature(value):
    byte_list = tf.train.BytesList(value = [value])
    return tf.train.Feature(bytes_list = byte_list)    

def _encode_image(current_image):
    _, encode_image = cv2.imencode('.jpg', current_image)
    return encode_image.tobytes()

def _prepare_image(image_path, scale_value):
    
    label = cv2.imread(image_path, cv2.IMREAD_COLOR)  
    height, width, _ = label.shape
    
    scaled_image = cv2.resize(label, dsize=(0, 0), fx=1./scale_value, 
                fy=1./scale_value, interpolation=cv2.INTER_NEAREST)
    image = cv2.resize(scaled_image, dsize=(0, 0), fx=scale_value,
                fy=scale_value, interpolation=cv2.INTER_NEAREST)    
    
    height, width, _ = min(image.shape,label.shape)
    image = image[0:height, 0:width]; label = label[0:height, 0:width]
    
    return image, label
    
def _create_sample(scale_value, image_path):
    
    image, label = _prepare_image(image_path, scale_value)
    encoded_image = _encode_image(image); encoded_label = _encode_image(label)
    features = dict(image = _bytes_feature(encoded_image), label = _bytes_feature(encoded_label))
    example = tf.train.Example(features=tf.train.Features(feature=features))
   
    return example.SerializeToString()
    
if __name__ == "__main__":
    
    scale_value = 2; record_folder = 'F:/SR_folder/record_folder'
    train_folder = 'F:/SR_folder/train_folder'; validation_folder = 'F:/SR_folder/validation_folder'
    
    train_list = glob.glob(train_folder + '/*.jpg')
    validation_list = glob.glob(validation_folder + '/*.jpg')
    
    input_set = (dict(image_list = validation_list, data_type = 'validation_dataset'),
                 dict(image_list = train_list, data_type = 'train_dataset'))
    
    if not os.path.exists(record_folder):
        os.mkdir(record_folder) 
    
    for item_set in input_set:
        
        samples_number = str(len(item_set['image_list']))
        record_name = item_set['data_type'] + '_' + str(scale_value) + '_' + samples_number + '.tfrecord'
        record_file = os.path.join(record_folder, record_name)
        record_writer = tf.io.TFRecordWriter(record_file)
        
        for image in item_set['image_list']: 
            serialized_example = _create_sample(scale_value, image)
            record_writer.write(serialized_example)
        
        print('File:', record_name, 'ready. -> Number of records:', samples_number)