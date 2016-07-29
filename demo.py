import caffe
import sys
#from utils.timer import Timer
#sys.path.append("/home/lijian/caffe-master/py-faster-rcnn/lib/")
#from fast_rcnn.test import im_detect
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
import os

def plot_detection(im_file,bbox_file,thresh,bbox_num,im_name):
  im = cv2.imread(im_file)
  im = cv2.resize(im,(500,350))
  im = im[:, :, (2, 1, 0)]
  fig, ax = plt.subplots(figsize=(12, 12))
  ax.imshow(im, aspect='equal')
  for cls in ["cat","dog"]:
          #print bbox_file+cls+".txt"
	  bbox = np.loadtxt(bbox_file+cls+".txt")
	  for i in range(0,len(bbox[:,1])):
	      if  bbox[i][1]<thresh or i>bbox_num:
		 break
              score=bbox[i][0+1]
	      x=bbox[i][0+2]
	      y=bbox[i][1+2]
	      w=bbox[i][2+2]
	      h=bbox[i][3+2]
	      ax.add_patch(plt.Rectangle((x, y),w - x,h - y, fill=False,edgecolor='red', linewidth=3.5))
              ax.text(x, y - 2,'{:s} {:.3f}'.format(cls, score),bbox=dict(facecolor='blue', alpha=0.5),fontsize=14, color='white')
  plt.axis('off')
  plt.tight_layout()
  plt.draw()
  print im_savefile+im_name
  savefig(im_savefile+im_name)


rootdir="/home/lijian/caffe/models/VGGNet/VOC0712/SSD_300x300/"
prototxt="/home/lijian/caffe/examples/ssd/testimage/deploy.prototxt"
caffemodel=rootdir+"VGG_VOC0712_SSD_300x300_iter_3352.caffemodel"
#im_file="/home/lijian/data/VOCdevkit/VOC2007/JPEGImages/"
im_file="/home/lijian/caffe/examples/ssd/testimage/pet_val/multi_pets/"
im_savefile="/home/lijian/caffe/examples/ssd/testimage/result/"
bbox_file="/home/lijian/caffe/examples/ssd/testimage/"
#testimagelist_file="/home/lijian/caffe/examples/ssd/testimage/dog_cat_test.txt"
#testimagelist_file=""
#img = cv2.imread(im_file)
#print dir(net.blobs['data'].data), net.blobs['data'].data.shape
"""
testimagelist=open(testimagelist_file)
imlist=[]
for line in testimagelist.readlines():
  line=line.rstrip('\n')
  imlist.append(line)

for im_num in range(1,10):
"""

for im_name in os.listdir(im_file):
        print im_name
	net = caffe.Net(prototxt, caffemodel, caffe.TEST)
	#im_path=im_file+imlist[im_num]+'.jpg'
        im_path=im_file+im_name
	print im_path
	#im_name=imlist[im_num]+'.jpg'
	#line=line.rstrip('\n')
	transformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape})
        print net.blobs['data'].data.shape
	transformer.set_transpose('data', (2,0,1))
	mean=np.array([104,117,123])
	transformer.set_mean('data',mean)
	transformer.set_raw_scale('data', 255)
	transformer.set_channel_swap('data', (2,1,0)) # if using RGB instead if BGR
	net.blobs['data'].data[...]= transformer.preprocess('data',caffe.io.load_image(im_path))
	net.forward()
	plot_detection(im_path,bbox_file,0.6,5,im_name)
	#plt.show()






