from settings import *
import numpy as np
from tfrecord_read_write import read_and_decode
import pickle

def calc_iou(box_a, box_b):
    """
	Calculate the Intersection Over Union of two boxes
	Each box specified by upper left corner and lower right corner:
	(x1, y1, x2, y2), where 1 denotes upper left corner, 2 denotes lower right corner

	Returns IOU value
	"""
	# Calculate intersection, i.e. area of overlap between the 2 boxes (could be 0)
	# http://math.stackexchange.com/a/99576
	# x_overlap = max(0, min(r1.right, r2.right)-max(r1.left, r2.left))
	#y_overlap = max(0, min(r1.bottom, r2.bottom) - max(r1.top, r2.top))
	#overlapArea = x_overlap * y_overlap

    #                     r1[right], r2[right]       r1[left],  r2[left]
    x_overlap = max(0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0]))
	
	#                    r1[bottom], r2[bottom]      r1[top],  r2[top]
    y_overlap = max(0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]))
    intersection = x_overlap * y_overlap  #Masahat area with overlap   مساحت ناحیه ی همپوشانی

	# Calculate union
	#Masahat r1   r1.Right - r1.Left      r1.Bottom - r1.Top    مساحت باکس a
    area_box_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])

	#Masahat r2   r2.Right  -  r2.Left    r1.Bottom - r1.Top   مساحت باکس b
    area_box_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
	# (A xor B)=> r1 xor r2
    union = area_box_a + area_box_b - intersection  #masahat area without overlap
	
    iou = np.divide(intersection , union, dtype=np.float) # مقدار همپوشانی باکس های a, b
	
    return iou

def find_gt_boxes(image_file, tag, gt_bboxes):
    
    signs_class = []
    signs_box_coords = []    
    for i in range(int(len(gt_bboxes)/4)):
        start_index, end_index = i*4, i*4+4
        signs_class.append(tag)
        scale = np.array([IMG_W, IMG_H, IMG_W, IMG_H])
        box_coords = np.divide(np.array(gt_bboxes[start_index:end_index]),scale, dtype=np.float)
        if(box_coords[0]==0. and box_coords[1]==0. and box_coords[2]==0. and box_coords[3]==0.):
            continue
        signs_box_coords.append(box_coords)

    y_ture_len = 0
    for fm_size in FM_SIZES:
        y_ture_len += fm_size[0] * fm_size[1]* NUM_DEFAULT_BOXES
    
    y_true_conf = np.zeros(y_ture_len)
    y_true_loc = np.zeros(y_ture_len*4)

    match_counter = 0
    for i, gt_box_coords in enumerate(signs_box_coords):
        y_true_idx = 0

        for fm_size in FM_SIZES:
            fm_h, fm_w = fm_size
            for row in range(fm_h):
                for col in range(fm_w):
                    for db in DEFAULT_BOXES:
                        x1_offset, y1_offset, x2_offset, y2_offset = db
                        abs_db_box_coords = np.array([
							max(0, col + x1_offset),
							max(0, row + y1_offset),
							min(fm_w, col+1 + x2_offset),
							min(fm_h, row+1 + y2_offset)
						])
                        '''abs_db_box_coords = np.array([
							max(0, col + col*x1_offset),
							max(0, row + row*y1_offset),
							min(fm_w, col+1 + col*x2_offset),
							min(fm_h, row+1 + row*y2_offset)
						])'''
                        scale = np.array([fm_w, fm_h, fm_w, fm_h])
                        db_box_coords = np.divide(abs_db_box_coords , scale, dtype=np.float)
                        if col==20 and row==20:
                            stop_check_point=0
                        iou = calc_iou(gt_box_coords, db_box_coords)

                        if iou >= IOU_THRESH:
    							# Update y_true_conf to reflect we found a match, and increment match_counter
                            y_true_conf[y_true_idx] = signs_class[i]
                            match_counter += 1

                            #print('(%d,%d)/'%(row,col))

							# مختصات پنجره یافته شده را نرمال کرده و در y_true_loc اضافه می کند
							# Calculate normalized box coordinates and update y_true_loc
                            abs_box_center = np.array([col + 0.5, row + 0.5])  # absolute coordinates of center of feature map cell
                            abs_gt_box_coords = gt_box_coords * scale  # absolute ground truth box coordinates (in feature map grid)
                            norm_box_coords = abs_gt_box_coords - np.concatenate((abs_box_center, abs_box_center))
                            y_true_loc[y_true_idx*4 : y_true_idx*4 + 4] = norm_box_coords

                        y_true_idx += 1

    return y_true_conf, y_true_loc, match_counter

def do_data_prep():    
    data_prep = {}

    filename_queue = tf.train.string_input_producer([TFRECORD_ADDRESS], num_epochs=1 )
    image, tag, bboxes = read_and_decode(TFRECORD_ADDRESS, filename_queue)
    init_op = tf.group(tf.global_variables_initializer(),
                tf.local_variables_initializer())
    
    with tf.Session() as sess:
        sess.run(init_op)
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)


        for i in range(TOTAL_DATA_TO_READ_FROM_TFRECORD):
            image_filename, label, gt_box_coords = sess.run([image, tag, bboxes]) #read_single_data_from_tfrecord(TFRECORD_ADDRESS)
            y_true_conf, y_true_loc, match_counter = find_gt_boxes(image_filename, label, gt_box_coords)
            if match_counter > 0:
                data_prep[image_filename] = {'y_true_conf': y_true_conf, 'y_true_loc': y_true_loc}
            print('%s/%s, matche_numbers: %s'%(i,614, match_counter))
        return data_prep

if __name__ == '__main__':
    data_prep = do_data_prep()

    with open('data+prep_%sx%s.p'%(IMG_W, IMG_H), 'wb') as f:
        pickle.dump(data_prep, f, protocol=2)
    
    print('Done. Saved prepared data to data_prep_%sx%s.p' % (IMG_W, IMG_H))
    print('Total images with >=1 matching box: %d' % len(data_prep.keys()))