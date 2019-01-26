'''
Data preparation
'''
from settings import *
import numpy as np
import pickle
import tensorflow as tf
from dataset_utils import ImageReader, _get_dataset_filename, image_to_tfexample, bytes_feature, int64_feature
import math
import sys
import os
from PIL import Image

def calc_iou(box_a, box_b):
    """
    Calculate the Intersection Over Union of two boxes
    Each box specified by upper left corner and lower right corner:
    (x1, y1, x2, y2), where 1 denotes upper left corner, 2 denotes lower right corner

    Returns IOU value
    """
    # Calculate intersection, i.e. area of overlap between the 2 boxes (could be 0)
    # http://math.stackexchange.com/a/99576
    x_overlap = max(0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0]))
    y_overlap = max(0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]))
    intersection = x_overlap * y_overlap

    # Calculate union
    area_box_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_box_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_box_a + area_box_b - intersection

    iou = intersection / union
    return iou


def find_gt_boxes(data_raw, image_file):
    """
    Given (global) feature map sizes, and single training example,
    find all default boxes that exceed Jaccard overlap threshold

    Returns y_true array that flags the matching default boxes with class ID (-1 means nothing there)
    """
    # Pre-process ground-truth data
    # Convert absolute coordinates to relative coordinates ranging from 0 to 1
    # Read the sign class label (note background class label is 0, sign labels are ints >=1)
    signs_data = data_raw[image_file]

    signs_class = []
    signs_box_coords = []  # relative coordinates
    for sign_data in signs_data:
        # Find class label
        sign_class = sign_data['class']
        signs_class.append(sign_class)

        # Calculate relative coordinates
        # (x1, y1, x2, y2), where 1 denotes upper left corner, 2 denotes lower right corner
        abs_box_coords = sign_data['box_coords']
        scale = np.array([IMG_W, IMG_H, IMG_W, IMG_H])
        box_coords = np.array(abs_box_coords) / scale
        signs_box_coords.append(box_coords)

    # Initialize y_true to all 0s (0 -> background)
    y_true_len = 0
    for fm_size in FM_SIZES:
        y_true_len += fm_size[0] * fm_size[1] * NUM_DEFAULT_BOXES
    y_true_conf = np.zeros(y_true_len)
    y_true_loc = np.zeros(y_true_len * 4)

    # For each GT box, for each feature map, for each feature map cell, for each default box:
    # 1) Calculate the Jaccard overlap (IOU) and annotate the class label
    # 2) Count how many box matches we got
    # 3) If we got a match, calculate normalized box coordinates and updte y_true_loc
    match_counter = 0
    for i, gt_box_coords in enumerate(signs_box_coords):
        y_true_idx = 0
        # for fm_idx, fm_size in enumerate(FM_SIZES):
        for fm_size in FM_SIZES:
            fm_h, fm_w = fm_size  # feature map height and width
            for row in range(fm_h):
                for col in range(fm_w):
                    for db in DEFAULT_BOXES:
                        # Calculate relative box coordinates for this default box
                        x1_offset, y1_offset, x2_offset, y2_offset = db
                        abs_db_box_coords = np.array([
                            max(0, col + x1_offset),
                            max(0, row + y1_offset),
                            min(fm_w, col+1 + x2_offset),
                            min(fm_h, row+1 + y2_offset)
                        ])
                        scale = np.array([fm_w, fm_h, fm_w, fm_h])
                        db_box_coords = abs_db_box_coords / scale

                        # Calculate Jaccard overlap (i.e. Intersection Over Union, IOU) of GT box and default box
                        iou = calc_iou(gt_box_coords, db_box_coords)

                        # If box matches, i.e. IOU threshold met
                        if iou >= IOU_THRESH:
                            # Update y_true_conf to reflect we found a match, and increment match_counter
                            y_true_conf[y_true_idx] = signs_class[i]
                            match_counter += 1

                            # Calculate normalized box coordinates and update y_true_loc
                            # absolute coordinates of center of feature map cell
                            abs_box_center = np.array([col + 0.5, row + 0.5])
                            # absolute ground truth box coordinates (in feature map grid)
                            abs_gt_box_coords = gt_box_coords * scale
                            norm_box_coords = abs_gt_box_coords - \
                                np.concatenate(
                                    (abs_box_center, abs_box_center))
                            y_true_loc[y_true_idx*4: y_true_idx *
                                       4 + 4] = norm_box_coords

                        y_true_idx += 1

    return y_true_conf, y_true_loc, match_counter


def do_data_prep_with_pickle(data_raw):
    data_prep = {}
    i = 0
    for image_file in data_raw.keys():

        y_true_conf, y_true_loc, match_counter = find_gt_boxes(
            data_raw, image_file)
        i += 1
        print('counter:%d - number of matchs:%d' % (i, match_counter))
        # Only want data points where we have matching default boxes
        if match_counter > 0:
            data_prep[image_file] = {
                'y_true_conf': y_true_conf, 'y_true_loc': y_true_loc}

    return data_prep


def do_data_prep_with_tfrecord(raw_data, tfrecord_filename, _NUM_SHARDS, dataset_directory_address, num_train_data, num_valid_data, split_name='train'):
    num__per_shard = math.ceil(len(raw_data)/float(_NUM_SHARDS))
    with tf.Graph().as_default():
        with tf.Session() as sess:

            end_index = 0
            shard_id = -1
            index = 0
            Vindex = 0  # this variable is used to compute number of data that are written in validation tfrecord

            # Change of the following loops in order to adapt it for repetition of train data and extract validation data
            # Both validation and train data gathering can be done in one run
            # repeat data for two time to ve
            # Change of the following loops in order to adapt it for repetition of train data and extract validation data
            # Both validation and train data gathering can be done in one run
            # repeat data for two time to verify in training
            for image_file in data_raw.keys():
                if (shard_id < 0 or index >= end_index+1) and index <= num_train_data:
                    shard_id += 1
                    end_index = min(
                        (shard_id+1) * num__per_shard, len(raw_data))
                    output_filename = _get_dataset_filename(
                        dataset_directory_address, split_name, shard_id, tfrecord_filename=tfrecord_filename, _NUM_SHARDS=_NUM_SHARDS)

                    tfrecord_writer_train = tf.python_io.TFRecordWriter(
                        output_filename)

                    y_true_conf, y_true_loc, match_counter = find_gt_boxes(
                        data_raw, image_file)
                    print('size of y_true_conf: %d' % (len(y_true_conf)))
                    print('size of y_true_loc: %d' % (len(y_true_loc)))

                    if(match_counter > 0):
                        image = Image.open('Caltech pedestrian dataset/Caltech pedestrian dataset/data_train/images_640x480/'+image_file[27:])
                        image = image.convert('L')
                        image.save('dataset/test/'+image_file[27:])

                        index += 1
                        if split_name == 'train':
                            index_angle = int(np.random.uniform(0, 5))
                            example = tf.train.Example(features=tf.train.Features(feature={
                                'image_address': bytes_feature(tf.compat.as_bytes(image_file)),
                                'y_true_conf': tf.train.Feature(float_list=tf.train.FloatList(value=y_true_conf.flatten())),
                                'y_true_loc': tf.train.Feature(float_list=tf.train.FloatList(value=y_true_loc.flatten())),
                                'index_angle': tf.train.Feature(int64_list=tf.train.Int64List(value=[index_angle]))
                            }))

                            tfrecord_writer_train.write(
                                example.SerializeToString())

                    sys.stdout.write(
                        '\r>> index: %d,   number of matchs: %d' % (index, match_counter))
                    sys.stdout.flush()

                else:

                        # dataset_directory_address + raw_data[i][0]
                    '''bboxes_coords = []
                                    for i, box in enumerate(raw_data[i][1]):
                                        bboxes_coords.append(box)'''
                    if(index <= num_train_data):
                        y_true_conf, y_true_loc, match_counter = find_gt_boxes(
                            data_raw, image_file)

                        if(match_counter > 0):
                            image = Image.open('Caltech pedestrian dataset/Caltech pedestrian dataset/data_train/images_640x480/'+image_file[27:])
                            image = image.convert('L')
                            image.save('dataset/test/'+image_file[27:])

                            index += 1

                            if split_name == 'train':
                                index_angle = int(np.random.uniform(0, 5))
                                example = tf.train.Example(features=tf.train.Features(feature={
                                    'image_address': bytes_feature(tf.compat.as_bytes(image_file)),
                                    'y_true_conf': tf.train.Feature(float_list=tf.train.FloatList(value=y_true_conf.flatten())),
                                    'y_true_loc': tf.train.Feature(float_list=tf.train.FloatList(value=y_true_loc.flatten())),
                                    'index_angle': tf.train.Feature(int64_list=tf.train.Int64List(value=[index_angle]))
                                }))

                                tfrecord_writer_train.write(
                                    example.SerializeToString())

                        sys.stdout.write(
                            '\r>> index: %d,  number of matchs: %d' % (index, match_counter))
                        sys.stdout.flush()

                    else:
                        if(Vindex == 0):
                            tfrecord_writer_train.close()
                            print('\nStart to prepare validation data\n')

                            split_name = 'validation'
                            output_filename = _get_dataset_filename(
                                dataset_directory_address, split_name, 0, tfrecord_filename=tfrecord_filename, _NUM_SHARDS=_NUM_SHARDS)
                            tfrecord_writer_validation = tf.python_io.TFRecordWriter(
                                output_filename)

                        y_true_conf, y_true_loc, match_counter = find_gt_boxes(
                            data_raw, image_file)

                        if(match_counter > 0):
                            image = Image.open('Caltech pedestrian dataset/Caltech pedestrian dataset/data_train/images_640x480/'+image_file[27:])                        
                            image = image.convert('L')
                            image.save('dataset/test/'+image_file[27:])
                            index += 1
                            Vindex += 1
                            if split_name == 'validation':
                                index_angle = int(np.random.uniform(0, 5))
                                example = tf.train.Example(features=tf.train.Features(feature={
                                    'image_address': bytes_feature(tf.compat.as_bytes(image_file)),
                                    'y_true_conf': tf.train.Feature(float_list=tf.train.FloatList(value=y_true_conf.flatten())),
                                    'y_true_loc': tf.train.Feature(float_list=tf.train.FloatList(value=y_true_loc.flatten())),
                                    'index_angle': tf.train.Feature(int64_list=tf.train.Int64List(value=[index_angle]))
                                }))

                                tfrecord_writer_validation.write(
                                    example.SerializeToString())

                                if(Vindex > num_valid_data):
                                    break

                        sys.stdout.write('\r>> index: %d,  number of matchs: %d' % (index, match_counter))
                        sys.stdout.flush()

            if(Vindex == 0):
                tfrecord_writer_train.close()
            if(Vindex > 0):
                tfrecord_writer_validation.close()
            print('number of validation data: %d' % (index-Vindex))


if __name__ == '__main__':
    #'data_raw_caltech_%sx%s-1500pics.p'
    #tfrecords_saved_model/ssd-rotation-v12-train_with_full_data-24806_trian_and_10631valid/data_raw_caltech_%sx%s-45914trainpics_v11.p' % (IMG_W, IMG_H) 
    with open('data_raw_caltech_640x480-45914trainpics_test.p', 'rb') as f:
        # with open('data_raw_caltech_640x480-1002pics.p', 'rb') as f:
        data_raw = pickle.load(f)

    print('Preparing data (i.e. matching boxes)')
    #data_prep = do_data_prep_with_pickle(data_raw)
    #do_data_prep_with_tfrecord(raw_data,    tfrecord_filename                                        , _NUM_SHARDS, dataset_directory_address,num_train_data, num_valid_data, split_name='train')
    do_data_prep_with_tfrecord(data_raw, DATASET_DIR+SUB_DIR+'tfrecord-%sx%ssize/tfrecord_v11' %
                               (IMG_W, IMG_H), 3, DATASET_DIR, 24806, 10631, split_name='train')
    '''do_data_prep_with_tfrecord(data_raw, DATASET_DIR+SUB_DIR+'tfrecord-%sx%ssize/tfrecord_v11' %
                               (IMG_W, IMG_H), 3, DATASET_DIR, 50, 50, split_name='train')'''
    
    # with open(DATASET_DIR+'data_prep_caltech_%sx%s-fulltrainpics.p' % (IMG_W, IMG_H), 'wb') as f:
    #	pickle.dump(data_prep, f)

    #print('Done. Saved prepared data to exteranal Hard with the name of data_prep_caltechDataset_%sx%s-4001trainpics.p' %
    print('Done. Saved tfrecord files to: '+DATASET_DIR+DATASET_DIR+SUB_DIR+'tfrecord-%sx%ssize/' %(IMG_W, IMG_H))
    # print('Total images with >=1 matching box: %d' % len(data_prep.keys()))
 
