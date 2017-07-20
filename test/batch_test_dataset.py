#!/usr/bin/env python
#coding: utf-8
import _init_paths
#import matplotlib.pyplot as plt
import numpy as np
import caffe, os, cv2
import argparse

import sys
reload(sys)
sys.setdefaultencoding('utf8')

#CLASSES = ('__background__',
#           'bus', 'car', 'person', 'truck','tricycle')

CLASSES = ('__background__', 'person')

#COLORS = {'__background__':(0,0,0),'bus':(255,0,0), 'car':(0,255,0), 'person':(0,0,255), 'truck':(255,255,0),'tricycle':(190,190,190)}
COLORS = {'__background__':(0,0,0), 'person':(255,0,0)}

CONF_THRESH = 0.5


# PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]], dtype=np.float32)
# PIXEL_STD = 0.017

# resnetV2
PIXEL_MEANS = np.array([[[128, 128, 128]]], dtype=np.float32)
PIXEL_STD = 0.0078125

# PIXEL_MEANS = np.array([[[103.94, 116.78, 123.68]]], dtype=np.float32)
# PIXEL_STD = 0.017

TEST_SCALE = 1000
MAX_SIZE = 1280

result_save_path = u"temp_test"

def _get_blob(im_orig, target_size=600, max_size=1000):
    """Converts an image into a network input.

    Arguments:
        im_orig (ndarray): a color image in BGR order
        target_size (int): desired size of short side
        max_size (int): max size of long side

    Returns:
        blob (ndarray): data blob holding an image
        im_scale (float32): image scale (relative to im_orig)
    """
    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    # im_orig = im_orig.astype(np.float32, copy=True)
    # im_orig -= PIXEL_MEANS
    # im_orig *= PIXEL_STD

    #im_scale = np.float32(target_size) / np.float32(im_size_min)
    im_scale = 1.0
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = np.float32(max_size) / np.float32(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)
    im = im.astype(np.float32, copy=True)
    im -= PIXEL_MEANS
    im *= PIXEL_STD

    height, width, channels = im.shape
    blob = im.reshape((1, height, width, channels)).transpose(0, 3, 1, 2)

    return blob, im_scale

def vis_detections(im, dets,class_name):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= 0.5)[0]
    global result_save_path


    with open(result_save_path + "//" + result_save_path + ".txt", "a+") as f:
        str_line = " " + class_name + " " + str(len(inds))
        f.write(str_line)

    if (dets.ndim == 1): # no detection
        return

    for i in xrange(dets.shape[0]):
        cls_ind = int(dets[i, 0])
        class_name = CLASSES[cls_ind]
        bbox = dets[i, 1:5]
        score = dets[i, -1]
        if score < CONF_THRESH:
            continue

        if (score >= 0.7):
            cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), COLORS[class_name], 3)
        elif (bbox[2] - bbox[0] < 40):
            cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 255), 3)
        else:
            pass
        cv2.putText(im, str(score), (int(bbox[0] + 30), int(bbox[1] + 30)), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
        with open(result_save_path + "//" + result_save_path + ".txt", "a+") as f:
            line_str = " " + str(score) + " " + str(int(bbox[0])) + " " + str(int(bbox[1])) + " " + str(int(bbox[2])) + " " + str(int(bbox[3]))
            # save the image
            # cv2.imwrite('./batch_test_result/model_1/20170720_1/part'+"_" + str(score) + "_" + str(int(bbox[0])) + "_" + str(int(bbox[1])) + "_" + str(int(bbox[2])) + "_" + str(int(bbox[3]))+'.jpg', im)
            f.write(line_str)

def demo(net, im_file):

    global result_save_path
    #im = cv2.imread(im_file.decode('gb2312').encode('utf8'))
    #im = cv2.imread(im_file.encode('utf8'))
    im = cv2.imread(im_file)
    if im == None:
        with open(result_save_path + "//" + result_save_path + "No_pic_list.txt", "a+") as f:
            f.write(im_file + "\n")
        return    

    # Detect all object classes and regress object bounds
    data, im_scale = _get_blob(im, TEST_SCALE, MAX_SIZE)
    im_info = np.array([[im.shape[0], im.shape[1], im_scale]], dtype=np.float32)
    forward_kwargs = {'data': data, 'im_info': im_info}
    net.blobs['data'].reshape(*(data.shape))

    blobs_out = net.forward(**forward_kwargs)

    with open(result_save_path + "//" + result_save_path + ".txt", "a+") as f:
        str_line = im_file.split("/")
        f.write(str_line[-1])

    dets = blobs_out["frcn_output"]

    # no detection
    if 1 == len(dets.shape):
        dets = np.array([[-1,0,0,0,0,0]], dtype=float)

    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        inds = np.where(dets[:, 0] == cls_ind)[0]
        dets1 = dets[inds] 
        vis_detections(im, dets1, cls)

    # inds = np.where(dets[:, 0] == 3)[0]
    # dets1 = dets[inds] 
    # vis_detections(im, dets1,"person")
    # inds = np.where(dets[:, 0] == 2)[0]
    # dets2 = dets[inds] 
    # vis_detections(im, dets2,"car")


    with open(result_save_path + "//" + result_save_path + ".txt", "a+") as f:
        f.write("\n")

    # save image as result
    file_name = im_file.split("/")[-2]
    #file_name = result_save_path + "/" + file_name.decode("gb2312")
    file_name = result_save_path + "/" + file_name

    if os.path.exists(file_name):
        pass
    else:
        os.mkdir(file_name)
    save_path = file_name
    try:
        os.mkdir(save_path)
    except:
        pass
    #dstpath = os.path.join(save_path, os.path.basename(im_file.decode("gb2312")))
    dstpath = os.path.join(save_path, os.path.basename(im_file))         
    rows, cols, channels = im.shape
    res = cv2.resize(im, (cols, rows))
    # res = cv2.resize(im, (cols , rows ))
    cv2.imwrite(dstpath, res)

if __name__ == '__main__':

    # Parse input arguments
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--prototxt', help='Network prototxt',
                        default='vgg16_test.prototxt', type=str)
    parser.add_argument('--caffemodel', help='Network model',
                        default='vgg16_faster_rcnn_iter_70000.caffemodel', type=str)
    #parser.add_argument('--image', help='image file', type=str)

    args = parser.parse_args()

    prototxt = args.prototxt
    caffemodel = args.caffemodel

    global result_save_path
    result_save_path =  u"test_model_fanjing_"
    result_save_path += str(MAX_SIZE)


    if not os.path.isfile(prototxt):
        raise IOError(('{:s} not found.\n').format(prototxt))
    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\n').format(caffemodel))

    if os.path.exists(result_save_path):
        pass
    else:
        os.mkdir(result_save_path)

    caffe.set_device(args.gpu_id)
    caffe.set_mode_gpu()

    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)
    hms_dataset = u"/home/luyi/share/data_sing/test_image_index_modify_part.txt"
    txt = open(hms_dataset, "r").readlines()

    for pic in txt:
        #pic = pic.decode("gb2312")
        pic = pic.strip()+'.jpg'
        #index = pic.rfind("_")
        #file = pic[:index]
        pic = "/home/luyi/share/data_sing/" + pic
        #print pic.decode("gb2312")
        #print pic.decode("utf-8")
        print pic
        #demo(net, pic.encode('utf8'))
        demo(net, pic)
    #raw_input()
