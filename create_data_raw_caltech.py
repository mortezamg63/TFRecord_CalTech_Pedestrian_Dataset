import os
import glob
import re
import json
from collections import defaultdict
from settings import *
import numpy as np
import tensorflow as tf
from dataset_utils import ImageReader, _get_dataset_filename, image_to_tfexample
from PIL import Image, ImageDraw
import math
import sys
import pickle

def read_caltech_annotation_json():
    annotations = json.load(open(DATASET_DIR+DATASET_DIR+SUB_DIR+'annotations.json'))

    '''out_dir = 'data/plots'
    f not os.path.exists(out_dir):
        os.makedirs(out_dir)'''

    img_fns = defaultdict(dict)

    for fn in sorted(glob.glob('dataset/images_640x480_v12/*.png')):#DATASET_DIR+SUB_DIR+'images/*.png')):
        set_name = re.search('(set[0-9]+)', fn).groups()[0]
        img_fns[set_name] = defaultdict(dict)

    for fn in sorted(glob.glob('dataset/images_640x480_v12/*.png')):#DATASET_DIR+SUB_DIR+'images/*.png')):
        set_name = re.search('(set[0-9]+)', fn).groups()[0]
        video_name = re.search('(V[0-9]+)', fn).groups()[0]
        img_fns[set_name][video_name] = []

    for fn in sorted(glob.glob('dataset/images_640x480_v12/*.png')):#DATASET_DIR + SUB_DIR + 'images/*.png')):
        set_name = re.search('(set[0-9]+)', fn).groups()[0]
        video_name = re.search('(V[0-9]+)', fn).groups()[0]
        n_frame = re.search('_([0-9]+)\.png', fn).groups()[0]
        img_fns[set_name][video_name].append((int(n_frame), fn))
    
    return annotations, img_fns


def do_data_prep():
    annotations, img_filenames = read_caltech_annotation_json()
    resized_dir = DATASET_DIR + DATASET_DIR + SUB_DIR + 'images_%sx%s/'%(IMG_W,IMG_H)
    signs = {'person':1,'person?':1 , 'person-fa':2, 'people':3}
    break_point = False
    data_raw = {}
    file_number = 0
    for set_name in sorted(img_filenames.keys()):
        if break_point:
            break
                    
        for video_name in sorted(img_filenames[set_name].keys()):
            if break_point:
                break
                        
            for frame_i, file_address in sorted(img_filenames[set_name][video_name]):
                sys.stdout.write('\r>>video: %s in set: %s--> frame_number:%d'%(video_name,set_name,frame_i))
                sys.stdout.flush()
                if str(frame_i) in annotations[set_name][video_name]['frames']:
                    image = Image.open(file_address)
                    orig_w, orig_h = image.size
                    image = image.convert('L')

                    if not os.path.exists(resized_dir):
                        os.makedirs(resized_dir)
                    
                    ##image = image.resize((IMG_W, IMG_H), Image.LANCZOS)
                    #image.save(os.path.join(resized_dir, file_address[74:]))                    
                                
                    x_scale = IMG_W / orig_w
                    y_scale = IMG_H / orig_h

                    bboxes_coords = []
                    tags = []
                    filename = file_address[re.match('^.*/set', file_address).regs[0][1]-3:]
                                
                    data = annotations[set_name][video_name]['frames'][str(frame_i)]
                    for datum in data:
                        ul_x, ul_y, w, h = [int(v) for v in datum['pos']]
                        br_x, br_y = ul_x+w, ul_y+h
                        new_box_coordinates = (ul_x*x_scale, ul_y*y_scale, br_x*x_scale, br_y*y_scale)
                        #new_box_coordinates = (ul_x, ul_y, br_x, br_y)
                        new_box_coordinates = [round(x) for x in new_box_coordinates]
                        bboxes_coords.append(new_box_coordinates)                    
                        tags.append(signs[datum['lbl']])
                        draw = ImageDraw.Draw(image)
                        draw.rectangle(((ul_x, ul_y),(br_x,br_y)))#,fill='white')
                    #image.show()
                    image.save(os.path.join(resized_dir, file_address[74:], filename))

                    the_list = []
                    for index in range(len(bboxes_coords)):
                        d = {'class': tags[index], 'box_coords': bboxes_coords[index]}
                        the_list.append(d)
                    data_raw[file_address] = the_list
                    file_number = file_number+1
                    if(file_number>10000000):
                        break_point=True
                        break
    with open('data_raw_caltech_%dx%d-%dtrainpics_test.p'%(IMG_W,IMG_H,file_number), 'wb') as f:
        pickle.dump(data_raw, f)



if __name__ == '__main__':
    do_data_prep()
    
