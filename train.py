import argparse
import Models , LoadBatches
import numpy as np
import pickle



parser = argparse.ArgumentParser()
parser.add_argument("--save_weights_path", type = str, default ="../data/crack_data_processed2/weights_fcn8/"  )
parser.add_argument("--train_images", type = str , default="../data/crack_data_processed2/train_320x480/" )
parser.add_argument("--train_annotations", type = str , default="../data/crack_data_processed2/trainannot_320x480/" )
parser.add_argument("--n_classes", type=int , default =9 )
parser.add_argument("--input_height", type=int , default = 320  )
parser.add_argument("--input_width", type=int , default = 480 )

parser.add_argument('--validate',action='store_false') # store_true: do not validate after training; store_false: validate after train.
parser.add_argument("--val_images", type = str , default = "../data/crack_data_processed2/val_320x480/")
parser.add_argument("--val_annotations", type = str , default = "../data/crack_data_processed2/valannot_320x480/")

parser.add_argument("--epochs", type = int, default = 50)
parser.add_argument("--batch_size", type = int, default = 5 )
parser.add_argument("--val_batch_size", type = int, default = 5 )
parser.add_argument("--load_weights", type = str , default = "") 
#../data/crack_data_processed2/weights_unet/.model.19
#../data/crack_data_processed2/weights_vgg_segnet/.model.19

parser.add_argument("--model_name", type = str , default = "fcn8")
parser.add_argument("--optimizer_name", type = str , default = "adadelta")


args = parser.parse_args()

train_images_path = args.train_images
train_segs_path = args.train_annotations
train_batch_size = args.batch_size
n_classes = args.n_classes
input_height = args.input_height
input_width = args.input_width
validate = args.validate
save_weights_path = args.save_weights_path
epochs = args.epochs
load_weights = args.load_weights

optimizer_name = args.optimizer_name
model_name = args.model_name

print "______________________________________________________________"
print "Model:", model_name
print "epochs: ", epochs
print "data path: ", train_images_path
print "_______________________________"

if validate:
	print "validate is true"
	val_images_path = args.val_images
	val_segs_path = args.val_annotations
	val_batch_size = args.val_batch_size

modelFns = { 'vgg_segnet':Models.VGGSegnet.VGGSegnet , 'vgg_unet':Models.VGGUnet.VGGUnet , 'vgg_unet2':Models.VGGUnet.VGGUnet2 , 'fcn8':Models.FCN8.FCN8 , 'fcn32':Models.FCN32.FCN32, 'segnet':Models.Segnet.segnet, 'unet':Models.Unet.Unet }
modelFN = modelFns[ model_name ]

m = modelFN( n_classes , input_height=input_height, input_width=input_width   )
m.compile(loss='categorical_crossentropy',
      optimizer= optimizer_name ,
      metrics=['accuracy'])


if len( load_weights ) > 0:
	m.load_weights(load_weights)


print "Model output shape" ,  m.output_shape

output_height = m.outputHeight
output_width = m.outputWidth

G  = LoadBatches.imageSegmentationGenerator( train_images_path , train_segs_path ,  train_batch_size,  n_classes , input_height , input_width , output_height , output_width   )


records = dict()
records.update({'acc': np.zeros((epochs,))})
records.update({'loss': np.zeros((epochs,))})
records.update({'val_acc': np.zeros((epochs,))})
records.update({'val_loss': np.zeros((epochs,))})


print "_______________________________"
print "Training starts..."

if validate:
	G2  = LoadBatches.imageSegmentationGenerator( val_images_path , val_segs_path ,  val_batch_size,  n_classes , input_height , input_width , output_height , output_width   )

if not validate:
	for ep in range( EPOCHS ):
		m.fit_generator( G , 512  , epochs=1 )
		m.save_weights( save_weights_path + "." + str( ep ) )
		m.save( save_weights_path + ".model." + str( ep ) )
else:
	for ep in range( epochs ):
		print ">> processing ep: ", ep        
		history = m.fit_generator( G , steps_per_epoch=20  , validation_data=G2 , validation_steps=20 ,  epochs=1, verbose =0 )
		m.save_weights( save_weights_path + "." + str( ep )  )
		m.save( save_weights_path + ".model." + str( ep ) )
		print(history.history.keys())
		print(history.history.values())
		records['acc'][ep] = history.history['acc'][-1]
		records['loss'][ep] = history.history['loss'][-1]
		records['val_acc'][ep] = history.history['val_acc'][-1]
		records['val_loss'][ep] = history.history['val_loss'][-1]

print(history.history.keys())
print(history.history.values())

with open(save_weights_path + model_name + '_Epo-'+ str(epochs) + '_history.pkl', 'wb') as f:
        pickle.dump(records, f)
