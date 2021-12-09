import numpy as np
import cv2
import matplotlib.pyplot as plt


def scaleUp(bbox,image):
  newbbox=np.copy(bbox)
  boxW = newbbox[2]
  boxH = newbbox[3]

  stepW = int(0.3 * boxW)
  stepH = int(0.3 * boxH)

  if newbbox[0] - stepW/2 >= 0 and newbbox[0] + boxW + stepW/2 <= image.shape[1]:
    newbbox[2]+=stepW
    newbbox[0] = newbbox[0] - stepW/2
  else:
    largestchange = max(min(newbbox[0], image.shape[1]-boxW-newbbox[0]),0)
    newbbox[2]= newbbox[2] + largestchange*2
    newbbox[0] = newbbox[0] - largestchange

  if newbbox[1] - stepH/2 >= 0 and newbbox[1] + boxH + stepH/2 <= image.shape[0]:
    newbbox[3]+=stepH
    newbbox[1] = newbbox[1] - stepH/2
  else:
    largestchange = max(min(newbbox[1], image.shape[0]-boxH-newbbox[1]),0)
    newbbox[3]= newbbox[3] + largestchange*2
    newbbox[1] = newbbox[1] - largestchange
  return newbbox



def scaleDown(bbox,image):
  newbbox=np.copy(bbox)
  boxW = newbbox[2]
  boxH = newbbox[3]

  stepW = int(0.3 * boxW)
  stepH = int(0.3 * boxH)

  newbbox[2]-=stepW
  newbbox[0] = newbbox[0] + stepW/2
  newbbox[3]-=stepH
  newbbox[1] = newbbox[1] + stepH/2
  return newbbox



def box_center_to_corner(bb):
  x=bb[0]
  y=bb[1]
  return (x,y),bb[2],bb[3]

def plotBoundingBox(img,bb,gt=None,saveCheck=None,directory=None):
  (x,y),w,h=box_center_to_corner(bb)
  fig, ax = plt.subplots()
  ax.imshow(img)
  rect=plt.Rectangle(xy=(x,y),width=w,height=h,fill=False,color="r")
  if gt:
    (x,y),w,h=box_center_to_corner(gt)
    gt_rect=plt.Rectangle(xy=(x,y),width=w,height=h,fill=False,color="b")
    ax.add_patch(gt_rect)
  ax.add_patch(rect)
  if saveCheck:
    plt.savefig(directory)
  else:
    plt.show()
  plt.close()


def calculate_IOU(box_1, box_2):
    b1_x, b1_y, b1_w, b1_h = box_1[0], box_1[1], box_1[2],box_1[3]
    b2_x, b2_y, b2_w, b2_h = box_2[0], box_2[1], box_2[2], box_2[3]

    b1_xmin, b1_ymin, b1_xmax, b1_ymax = box_1[0]-int(box_1[2]/2), box_1[1]-int(box_1[3]/2),box_1[0]-int(box_1[2]/2) + box_1[2], box_1[1]-int(box_1[3]/2) + box_1[3] # x, y, x+w = x_max, y + w = y_max
    b2_xmin, b2_ymin, b2_xmax, b2_ymax=  box_2[0]-int(box_2[2]/2), box_2[1]-int(box_2[3]/2),box_2[0]-int(box_2[2]/2) + box_2[2], box_2[1]-int(box_2[3]/2) + box_2[3]
           
    i_x1 = max(b1_xmin, b2_xmin)
    i_y1 = max(b1_ymin, b2_ymin)
    i_x2 = min(b1_xmax, b2_xmax)
    i_y2 = min(b1_ymax, b2_ymax)
        
    intersection_area = max(0, i_x2 - i_x1) * max(0, i_y2 - i_y1)
    box1_area = (b1_xmax - b1_xmin) * (b1_ymax - b1_ymin)
    box2_area = (b2_xmax - b2_xmin) * (b2_ymax - b2_ymin)

    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

def get_gt_values_VOT(gt_path):
  with open(gt_path, 'r') as f:
    lines = f.readlines()
    boxes = []
    for line in lines:
      if not line.strip():
        continue
      counter=0
      l=[0,0,0,0]
      for x in line.rstrip('\n').split('.0,'):
        if x.find(".0")!=-1:
          x=x.replace(".0","")
        l[counter]=int(x)
        counter+=1
        if counter==4:
          boxes.append(l)
    return boxes

def get_gt_values_OTB(gt_path):
  with open(gt_path, 'r') as f:
      lines = f.readlines()
      boxes = []
      for line in lines:
          if not line.strip():
              continue
          x, y, w, h = [int(x) for x in line.split(',')]
          boxes.append([x, y, w, h])
      return boxes




