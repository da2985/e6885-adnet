import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
import time
import numpy as np
import scipy.io as sio
import cv2, glob, os, re
import random
import linecache





data_home = 'trainframes/'
#LIST OF DATASET PATHS ['basepath/BlurBody', 'basepath/Basketball', etc ]
dataset = glob.glob(os.path.join(data_home,'*'))
number_of_sequences_to_train = 1
k=10
print(dataset)
random.shuffle(dataset)
#choose one video for training
for v, vd in enumerate(dataset[0:]):
    if (v == number_of_sequences_to_train):
        break  
    ground_truth = open('%s/groundtruth_rect.txt'%vd)
    frames = glob.glob(os.path.join('%s/img'%vd, '*.jpg'))
    #Sample k video clips (between the given ground truths)
    number_of_frames = len(frames)
    start_frame = random.randint(0, number_of_frames-k-1) 
    print(start_frame)
    end_frame = start_frame + k
    #paths of img frames to train
    train_sequence = frames[start_frame:end_frame]
    #list of k arrays with the 4 numbers of the ground_truth rect
    ground_truths = []
    for i in range(k):
        line = linecache.getline('%s/groundtruth_rect.txt'%vd, start_frame + i + 1)
        frame_gt = np.array(re.findall('\d+',line),dtype=int)
        ground_truths.append(frame_gt)
    print(train_sequence)
    print(ground_truths)
        

