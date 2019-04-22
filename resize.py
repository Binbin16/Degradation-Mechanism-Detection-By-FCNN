from PIL import Image
import os

'''
inputPath = "../data/crack_data_processed2/train_cp1/"
outPath = "../data/crack_data_processed2/train_320x480/"


def processImage(filesource, destsource, name):

    im = Image.open(filesource + name)
    im = im.resize((320, 480))
    im.save(destsource + name)


def run():

    os.chdir(inputPath)
    for i in os.listdir(os.getcwd()):
        processImage(inputPath, outPath, i)
run()
'''
path = "../data/crack_data_processed2/val_320x480/"
dirs = os.listdir( path )

def resize():
    for item in dirs:
        print item
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            imResize = im.resize((480,320))
            imResize.save(f + '.png')

resize()
