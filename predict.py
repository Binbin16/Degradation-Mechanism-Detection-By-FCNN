import argparse
import Models , LoadBatches
from keras.models import load_model
import glob
import cv2
import numpy as np
import random

parser = argparse.ArgumentParser()
parser.add_argument("--save_weights_path", type = str, default ="../data/crack_data_processed2/weights_fcn8/") #"../data/dataset1/weights/"   
parser.add_argument("--epoch_number", type = int, default = 49 ) ## 
parser.add_argument("--test_images", type = str , default = "../data/crack_data_processed2/test_320x480/") 
parser.add_argument("--test_annotations", type = str , default = "../data/crack_data_processed2/testannot_320x480/") #"../data/dataset1/images_prepped_test/"
parser.add_argument("--output_path", type = str , default = "../data/crack_data_processed2/predictions_fcn8/") #"../data/dataset1/predictions/"
parser.add_argument("--input_height", type=int , default = 320  )
parser.add_argument("--input_width", type=int , default = 480 )
parser.add_argument("--model_name", type = str , default = "fcn8")
parser.add_argument("--n_classes", type=int , default =9 )

args = parser.parse_args()

n_classes = args.n_classes
model_name = args.model_name
images_path = args.test_images
segs_path = args.test_annotations
input_width =  args.input_width
input_height = args.input_height
epoch_number = args.epoch_number

print "_____________________________________________________________________________________________"
print "Model:", model_name
print "Load weights from epochs: ", args.save_weights_path + "." + str(  epoch_number )
print "data path: ", images_path
print "_______________________________"


modelFns = { 'vgg_segnet':Models.VGGSegnet.VGGSegnet , 'vgg_unet':Models.VGGUnet.VGGUnet , 'vgg_unet2':Models.VGGUnet.VGGUnet2 , 'fcn8':Models.FCN8.FCN8 , 'fcn32':Models.FCN32.FCN32, 'segnet':Models.Segnet.segnet, 'unet':Models.Unet.Unet    }
modelFN = modelFns[ model_name ]

m = modelFN( n_classes , input_height=input_height, input_width=input_width   )
m.load_weights(  args.save_weights_path + "." + str(  epoch_number )  )
m.compile(loss='categorical_crossentropy',
      optimizer= 'adadelta' ,
      metrics=['accuracy'])


output_height = m.outputHeight
output_width = m.outputWidth

images = glob.glob( images_path + "*.jpg"  ) + glob.glob( images_path + "*.png"  ) +  glob.glob( images_path + "*.jpeg"  )
images.sort()

#colors = [  ( random.randint(0,255),random.randint(0,255),random.randint(0,255)   ) for _ in range(n_classes)  ]
background = [255,255,255] #white
noCracking = [128,0,0] # dark red
parallel= [0,255,0] #green
mudflat = [255,69,0] # red
localized = [128,64,128] # dark purple
delamination = [60,40,222] # blue
transverseBranching = [204,204,0] # dark yellow
longitudialBraching = [255,128,0] # orange
blistering = [255,0,0] # red
colors = [background,noCracking, parallel ,mudflat, localized, delamination, transverseBranching, longitudialBraching, blistering ]

predictions =np.empty((len(images),output_height ,  output_width , n_classes))

for imgName in images:
	outName = imgName.replace( images_path ,  args.output_path )
	X = LoadBatches.getImageArr(imgName , args.input_width  , args.input_height  )
	pr = m.predict( np.array([X]) )[0]
	#pr = pr.reshape(( output_height ,  output_width , n_classes ) ).argmax( axis=2 )
	pr = pr.reshape(( output_height ,  output_width , n_classes ) ) # replaced 62 to 63+64
	pr1 = pr.argmax( axis=2 ) 
	predictions[images.index(imgName), :, :, :] = pr    
	seg_img = np.zeros( ( output_height , output_width , 3  ) )
	for c in range(n_classes):
		seg_img[:,:,0] += ( (pr1[:,: ] == c )*( colors[c][0] )).astype('uint8')
		seg_img[:,:,1] += ((pr1[:,: ] == c )*( colors[c][1] )).astype('uint8')
		seg_img[:,:,2] += ((pr1[:,: ] == c )*( colors[c][2] )).astype('uint8')
	seg_img = cv2.resize(seg_img  , (input_width , input_height ))
	cv2.imwrite(  outName , seg_img )
    

# get real labels, to compare with predictions later.
segmentations  = glob.glob( segs_path + "*.jpg"  ) + glob.glob( segs_path + "*.png"  ) +  glob.glob( segs_path + "*.jpeg"  )
segmentations.sort()
label_tensor = np.empty((len(segmentations),output_height ,  output_width ))
for seg in segmentations:
	#print seg
	segArr = LoadBatches.getSegmentationArr( seg , n_classes , output_width , output_height )
	#print segArr.shape
	segArr = segArr.reshape(( output_height ,  output_width , n_classes ) )
	segArr1 = segArr.argmax( axis=2 )
	#print segArr.shape, segArr1.shape
	label_tensor[segmentations.index(seg), :, :] = segArr1    

    
##### To print out per class accuracy
def per_class_acc(predictions, label_tensor):
	labels = label_tensor
	size = predictions.shape[0]
	print size
	num_class = predictions.shape[3]
	hist = np.zeros((num_class, num_class))
	for i in range(size):
		hist += fast_hist(labels[i].flatten(), predictions[i].argmax(2).flatten(), num_class)
	acc_total = np.diag(hist).sum() / hist.sum()
	print ('accuracy = %f'%np.nanmean(acc_total))
	iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
	print ('mean IU  = %f'%np.nanmean(iu))
	for ii in range(num_class):
		if float(hist.sum(1)[ii]) == 0:
			acc = 0.0
		else:
			acc = np.diag(hist)[ii] / float(hist.sum(1)[ii])
		print("    class # %d accuracy = %f "%(ii,acc))
        
def fast_hist(a, b, n):
	k = (a >= 0) & (a < n)
	return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

## print out the per class accuracy
per_class_acc(predictions, label_tensor)
