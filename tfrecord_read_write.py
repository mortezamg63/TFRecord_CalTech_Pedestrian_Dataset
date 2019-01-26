import re
from PIL import Image, ImageDraw
import tensorflow as tf
import math
from dataset_utils import ImageReader, _get_dataset_filename, image_to_tfexample
import numpy as np
from scipy.ndimage.interpolation import rotate
#import gdspy as gd
import sys
import matplotlib.pyplot as plt
from settings import ROTATIONS,IMG_H,IMG_W, BATCH_SIZE
from dataset_utils import int64_feature, bytes_feature
from settings import *
import os




# this function read filename and box corners for each object
def read_content_annotation(annotation_file):    
    

    f = open(file=annotation_file, encoding='ISO-8859-1')

    filename=''    

    box_coordinates = []
    for line in f:
        line= str(line)
        srch_filename = re.match('^.*filename :', line)    
        srch_box1 = re.match('^.*(Xmax, Ymax)', line)
        if srch_filename != None:
            index_end_str = srch_filename.regs[0][1]
            filename = (line[index_end_str+2:])[:-2]
        elif srch_box1 != None:
            index_end_str = srch_box1.regs[0][1]
            #xmin, ymin, xmax, ymax = [int(num) for num in re.findall(r'\d+',line[index_end_str:])]
            box = [int(num) for num in re.findall(r'\d+',line[index_end_str:])]
            box_coordinates.append(box)                 
    
    return [filename, box_coordinates]

# This function get data and gather them in raw_data variable
def gathering_data( annotation_list_file_address, dataset_directory_address, encode='ISO-8859-1'):
    raw_data=[]

    

    with open(annotation_list_file_address, encoding=encode) as annot_file:
        for line in annot_file:
            data = read_content_annotation(dataset_directory_address + line[:-1])
            raw_data.append(data)

    return raw_data

# Function for rotating points in rotated boxes. But the function does not work correctly. It needs to pay more 
# attention in upcoming works
def new_position_of_point(x,y,cx,cy,theta):
    theta = theta * math.pi / 180
    new_x = (x*math.cos(theta)+y*math.sin(theta))
    new_y = (y*math.cos(theta)-x*math.sin(theta))
    return int(new_x), int(new_y)
    

# all gathered/collected data are saved in tfrecordfile in this function. Also images are read and resized
# here. Then boxes are changed based on alters in size because dataset images do not have the same size
# Here all boxes must be rotated and the corner points must be computed after rotation
def data_prep(raw_data, tfrecord_filename, _NUM_SHARDS,
    dataset_directory_address, split_name='Train'):
    # raw_data[i][0] image address for ith index 
    # raw_data[i][1] boxes in image with ith index
    xMax =0#-----------------------
    yMax =0#-----------------------
    xMin =0#-----------------------
    yMin = 0#-----------------------
    areaMax =0#-----------------------
    areaMin = 0#-----------------------
    maxArea_coords =[0,0,0,0]#-----------------------
    minArea_coords = [0,0,0,0]#-----------------------

    resize_dir = 'resized_images_INRIA_%dx%d/'%(TARGET_W, TARGET_H)
    num__per_shard = math.ceil(len(raw_data)/float(_NUM_SHARDS))
    with tf.Graph().as_default():
      #image_reader = ImageReader()

      with tf.Session() as sess:
        for shard_id in range(_NUM_SHARDS):
          output_filename = _get_dataset_filename(dataset_directory_address, split_name, 
                            shard_id, tfrecord_filename=tfrecord_filename, _NUM_SHARDS=_NUM_SHARDS)
                
          with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
            start_index = shard_id * num__per_shard
            end_index = min((shard_id+1) * num__per_shard, len(raw_data))
            for i in range(start_index, end_index):                
              image = Image.open(dataset_directory_address + raw_data[i][0])
              image_address = os.path.join(resize_dir,raw_data[i][0][10:])#dataset_directory_address + raw_data[i][0]
              sys.stdout.write('\r>>Converting image %d/%d shard %d' % (i+1, len(raw_data), shard_id))
              sys.stdout.flush()
              
              orig_w, orig_h = image.size
              image = image.convert('L') # 8-bit grayscale
              image = image.resize((IMG_W, IMG_H), Image.LANCZOS) # high-quality downsampling filter
              if not os.path.exists(resize_dir):
                  os.makedirs(resize_dir)
                
              image.save(os.path.join(resize_dir, raw_data[i][0][10:]))

              x_scale = IMG_W / orig_w
              y_scale = IMG_H / orig_h
              bboxes_coords = []              
              #for box in raw_data[i][1]:
              for i, box in enumerate(raw_data[i][1]):
                ul_x, ul_y, br_x, br_y = box
                new_box_coordinates = (ul_x*x_scale, ul_y*y_scale, br_x*x_scale, br_y*y_scale)
                new_box_coordinates = [round(x) for x in new_box_coordinates]                
                    
                bboxes_coords.append(new_box_coordinates)                
                "raw_data[i][1][0][0] = new_box_coordinates[0]"
                "raw_data[i][1][0][1] = new_box_coordinates[1]"
                "raw_data[i][1][0][2] = new_box_coordinates[2]"
                "raw_data[i][1][0][3] = new_box_coordinates[3]"

              #--------------------- create tfexample -----------
              b1_ulx = b1_uly = b1_brx = b1_bry = b2_ulx = b2_uly = b2_brx = b2_bry = b3_ulx = b3_uly = b3_brx = b3_bry = b4_ulx = b4_uly = b4_brx = b4_bry = 0
              for i , box in enumerate(bboxes_coords):                  
                if i == 0:
                    b1_ulx , b1_uly , b1_brx , b1_bry = box
                elif i ==1:
                    b2_ulx , b2_uly , b2_brx , b2_bry = box
                elif i==2:
                    b3_ulx , b3_uly , b3_brx , b3_bry = box
                elif i==3:
                    b4_ulx , b4_uly , b4_brx , b4_bry = box
                '''#  ----------- statistics ------------------
                    if(box[0]==0 and box[1]==0 and box[2]==0 and box[3]==0):
                        continue
                    xMax = max(xMax, box[2])
                    yMax = max(yMax, box[3])
                    
                    if xMin==0:
                        xMin = box[0]
                        yMin = box[1]
                    else:
                        xMin = min(xMin, box[0])
                        yMin = min(yMin, box[1])
                    
                    Area = (box[2] - box[0]) * (box[3] - box[1])
                                                        
                    if areaMax==0:
                        areaMax = Area
                        maxArea_coords = box
                    elif Area>areaMax:
                        areaMax = Area
                        maxArea_coords=box
                    
                    if areaMin==0:
                        areaMin = Area
                        minArea_coords=box
                    elif Area<areaMin:
                        areaMin = Area
                        minArea_coords=box'''

              example = tf.train.Example(features=tf.train.Features(feature={
                        #'imagedata': bytes_feature(image_data.tostring()),
                        'image_address': bytes_feature(tf.compat.as_bytes(image_address)),
                        'tag': int64_feature(1),
                        'box1_x0': int64_feature(b1_ulx),
                        'box1_y0': int64_feature(b1_uly),
                        'box1_x1': int64_feature(b1_brx),
                        'box1_y1': int64_feature(b1_bry),
                        'box2_x0': int64_feature(b2_ulx),
                        'box2_y0': int64_feature(b2_uly),
                        'box2_x1': int64_feature(b2_brx),
                        'box2_y1': int64_feature(b2_bry), 
                        'box3_x0': int64_feature(b3_ulx),
                        'box3_y0': int64_feature(b3_uly),
                        'box3_x1': int64_feature(b3_brx),
                        'box3_y1': int64_feature(b3_bry),
                        'box4_x0': int64_feature(b4_ulx),
                        'box4_y0': int64_feature(b4_uly),
                        'box4_x1': int64_feature(b4_brx),
                        'box4_y1': int64_feature(b4_bry),
                    }))   
              '''# Rotating resized image and its boxes
                    #draw = ImageDraw.Draw(image)              
                    cx = int(((new_box_coordinates[2]-new_box_coordinates[0])/2)+new_box_coordinates[0])
                    cy = int(((new_box_coordinates[3]-new_box_coordinates[1])/2)+new_box_coordinates[1])
                    #  x0y1-------x0y3
                    #    |         |
                    #    |  cx,cy  |
                    #    |         |
                    #  x2,y1-----x2,y3
                    x0, y1, x2, y3 = new_box_coordinates[0], new_box_coordinates[1], new_box_coordinates[2], new_box_coordinates[3]            

                    angle=45
                    new_img = rotate(image, angle, reshape=False)
                    pil_img = Image.fromarray(new_img)
                    
                    polygon = gd.Rectangle((x0,y1),(x2,y3))
                    polygon.rotate(angle*math.pi/180,center=(cx,cy))
                    p0, p1, p2, p3 = polygon.points[0], polygon.points[1], polygon.points[2], polygon.points[3]
                    #  x0y0-------x1y1=p1
                    #    |         |
                    #    |  cx,cy  |
                    #    |         |
                    #  x3,y3-----x2,y2=p2
                    def bound_limitation(x,max_x):
                        if(x<0):
                        x=0
                        elif(x>max_x):
                        x = max_x
                        return x
                    
                    x0 = bound_limitation(p0[0],target_img_width)
                    x1 = bound_limitation(p1[0],target_img_width)
                    x2 = bound_limitation(p2[0],target_img_width)
                    x3 = bound_limitation(p3[0],target_img_width)
                    y0 = bound_limitation(p0[1],target_img_height)
                    y1 = bound_limitation(p1[1],target_img_height)
                    y2 = bound_limitation(p2[1],target_img_height)
                    y3 = bound_limitation(p3[1],target_img_height)

                    draw = ImageDraw.Draw(pil_img)
                    draw.line(((x0,y0),(x1,y1)))
                    draw.line(((x1,y1),(x2,y2)))
                    draw.line(((x2,y2),(x3,y3)))
                    draw.line(((x3,y3),(x0,y0)))
                    draw.rectangle((cx-3,cy-3,(cx+3,cy+3)), fill='white')
                    pil_img.show()'''

              "image = np.asarray(image)"
              "images = np.array([image])"

              "images = np.expand_dims(images, axis=-1)" ## need extra dimension of size 1 for grayscale
              # ROTATING IMAGE IN DIFFERENT ANGLES AND ADD TO THE LAST COLUMN
              "tiled = np.tile(np.expand_dims(images, 4), [len(ROTATIONS)])"              
              #angles=[0]
              #for transformation_index, angle in enumerate(angles):
              #  tiled[:,:,:,:, transformation_index] = rotate(tiled[:, :, :, :, transformation_index], angle, axes=[1, 2], reshape=False)'''
              "example = image_to_tfexample(tiled,new_box_coordinates,1)" # class_id = 1 is pedestrian
              #example = image_to_tfexample(image_address,new_box_coordinates,1)              
              tfrecord_writer.write(example.SerializeToString())
            tfrecord_writer.close()
    sys.stdout.write('\n')            
    sys.stdout.flush()    



# defining features for getting them from saved tfrecord file
def read_and_decode(tfrecord_filename,filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(serialized_example,
                                        features = {                                            
                                            'box1_x0': tf.FixedLenFeature([],tf.int64),
                                            'box1_y0': tf.FixedLenFeature([],tf.int64),
                                            'box1_x1': tf.FixedLenFeature([],tf.int64),
                                            'box1_y1': tf.FixedLenFeature([],tf.int64),

                                            'box2_x0': tf.FixedLenFeature([],tf.int64),
                                            'box2_y0': tf.FixedLenFeature([],tf.int64),
                                            'box2_x1': tf.FixedLenFeature([],tf.int64),
                                            'box2_y1': tf.FixedLenFeature([],tf.int64),

                                            'box3_x0': tf.FixedLenFeature([],tf.int64),
                                            'box3_y0': tf.FixedLenFeature([],tf.int64),
                                            'box3_x1': tf.FixedLenFeature([],tf.int64),
                                            'box3_y1': tf.FixedLenFeature([],tf.int64),

                                            'box4_x0': tf.FixedLenFeature([],tf.int64),
                                            'box4_y0': tf.FixedLenFeature([],tf.int64),
                                            'box4_x1': tf.FixedLenFeature([],tf.int64),
                                            'box4_y1': tf.FixedLenFeature([],tf.int64),
                                            #'bboxes': tf.FixedLenFeature([],tf.int64),
                                            'tag': tf.FixedLenFeature([],tf.int64),
                                            'image_address': tf.FixedLenFeature([], tf.string)
                                        })

    #image = tf.decode_raw(features['image'],tf.uint8)# this is changed from tf.uint8 to tf.float32
    
    box1_x0 = tf.cast(features['box1_x0'], tf.int64)
    box1_y0 = tf.cast(features['box1_y0'], tf.int64)
    box1_x1 = tf.cast(features['box1_x1'], tf.int64)
    box1_y1 = tf.cast(features['box1_y1'], tf.int64)

    box2_x0 = tf.cast(features['box2_x0'], tf.int64)
    box2_y0 = tf.cast(features['box2_y0'], tf.int64)
    box2_x1 = tf.cast(features['box2_x1'], tf.int64)
    box2_y1 = tf.cast(features['box2_y1'], tf.int64)

    box3_x0 = tf.cast(features['box3_x0'], tf.int64)
    box3_y0 = tf.cast(features['box3_y0'], tf.int64)
    box3_x1 = tf.cast(features['box3_x1'], tf.int64)
    box3_y1 = tf.cast(features['box3_y1'], tf.int64)

    box4_x0 = tf.cast(features['box4_x0'], tf.int64)
    box4_y0 = tf.cast(features['box4_y0'], tf.int64)
    box4_x1 = tf.cast(features['box4_x1'], tf.int64)
    box4_y1 = tf.cast(features['box4_y1'], tf.int64)
    #bboxes = tf.cast(features['bboxes'],tf.int32)
    tag    = tf.cast(features['tag'], tf.int64)   
    image_address = tf.cast(features['image_address'], tf.string)

    #tag = tf.cast(features['tag'], tf.int32)
    #image_shape = tf.stack([image_height, image_width, 1, len(angles)])#   because height and width are gotten from tfrecord file, tf.train.shuffle can not work.
    #image_shape = tf.stack([IMG_H, IMG_W, 1, 1])
    #image = tf.reshape(image, image_shape)
    #image = tf.image.resize_images(image,(image_height, image_width))
    #image_size_const = tf.constant((image_height, image_width,1,1), dtype=tf.int32)
    '''# Random transformations can be put here: right before you crop images
    # to predefined size. To get more information look at the stackoverflow
    # question linked above.
    
    resized_image = tf.image.resize_image_with_crop_or_pad(image=image,
                                           target_height=IMAGE_HEIGHT,
                                           target_width=IMAGE_WIDTH)'''
    return image_address, tag, [box1_x0, box1_y0, box1_x1, box1_y1, box2_x0, box2_y0, box2_x1, box2_y1, box3_x0, box3_y0, box3_x1, box3_y1, box4_x0, box4_y0, box4_x1, box4_y1]

# read tfrecord file and return data as batch
def read_batch_from_tfrecord(tfrecorder_filename):
    filename_queue = tf.train.string_input_producer([tfrecorder_filename], num_epochs=1 )
    image, tag, box_x0, box_y0, box_x1, box_y1 = read_and_decode(tfrecorder_filename, filename_queue)
    #  batch reading   

    # single reading
    init_op = tf.group(tf.global_variables_initializer(),
                tf.local_variables_initializer())
    
    with tf.Session() as sess:
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        image, tag, box_x0, box_y0, box_x1, box_y1 = tf.train.shuffle_batch([image, tag, box_x0, box_y0, box_x1, box_y1], batch_size=BATCH_SIZE, capacity=1000, num_threads=3, min_after_dequeue=10)
        
        # Creates batches by randomly shuffling tensors        
        
        arr1 = np.asarray(image[0].eval())
        arr1 = arr1[:,:,0]
        pil_img = Image.fromarray(arr1[:,:,0])
        pil_img.show()
        # single image and its data reading'''
        
        '''for i in range(6):
            img, label, bx_x0, bx_y0, bx_x1, bx_y1 = sess.run([image, tag, box_x0, box_y0, box_x1, box_y1])
            # Convert to numpy array and next to  PIL Image
            arr= np.asarray(img[:,:,:,0])
            arr = arr[:,:,0]
            pil_img=Image.fromarray(arr)
            pil_img.show() # show pil image
            plt.imshow(arr[:,:,0], cmap='gray')
            plt.show() # show numpy array image'''

        
        coord.request_stop()
        coord.join(threads)

        return image, tag, [box_x0, box_y0, box_x1, box_y1]

def read_single_data_from_tfrecord(tfrecorder_filename):
    filename_queue = tf.train.string_input_producer([tfrecorder_filename], num_epochs=1 )
    #image, tag, box_x0, box_y0, box_x1, box_y1 = read_and_decode(tfrecorder_filename, filename_queue)
    image, tag, bboxes = read_and_decode(tfrecorder_filename, filename_queue)
    init_op = tf.group(tf.global_variables_initializer(),
                tf.local_variables_initializer())
    
    with tf.Session() as sess:
        sess.run(init_op)
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)   
        for i in range(3):     
            #img, label, bx_x0, bx_y0, bx_x1, bx_y1 = sess.run([image, tag, box_x0, box_y0, box_x1, box_y1])
            img, label, boxes = sess.run([image, tag, bboxes])
            # Convert to numpy array and next to  PIL Image
            """arr= np.asarray(img[:,:,:,0])
            arr = arr[:,:,0]
            pil_img=Image.fromarray(arr)
            draw = ImageDraw.Draw(pil_img)
            draw.rectangle(((bx_x0,bx_y0),(bx_x1,bx_y1)))        
            pil_img.show()"""
                
            """coord.request_stop()
            coord.join(threads)"""


    return img, label, bboxes #[bx_x0, bx_y0, bx_x1, bx_y1]

        





if __name__ == '__main__':
    
    #raw_data = gathering_data('INRIA_Dataset/INRIAPerson/Train/annotations.lst', 'INRIA_Dataset/INRIAPerson/')
    #data_prep(raw_data,'mydataset',1,'INRIA_Dataset/INRIAPerson/', split_name='Train')
    #read_batch_from_tfrecord('INRIA_Dataset/INRIAPerson/mydataset_Train_00000-of-00001.tfrecord')
    img1, tag1,boxes1 = read_single_data_from_tfrecord(DATASET_DIR+'data_train/tfrecord_caltechDataset_train.tfrecord') #'INRIA_Dataset/INRIAPerson/mydataset_Train_00000-of-00001.tfrecord')    