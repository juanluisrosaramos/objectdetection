import tensorflow as tf
import os
import io
import xmltodict
import pprint
import json
from object_detection.utils import dataset_util
from collections import defaultdict
from lxml import objectify
from PIL import Image
import PIL.Image
from collections import Counter

rootDir = 'images'
tfRecordName = 'test.tfrecord'
 
flags = tf.app.flags
flags.DEFINE_string('output_path', tfRecordName, 'Path to output TFRecord')
FLAGS = flags.FLAGS
json_file = 'project3.json'
with open(json_file, "r") as read_file:
     data = json.load(read_file)
labels_in_dataset = []


def get_cordinates(filename):
    BadGums = ''
    GoodGums = ''
    GoodTeeths = ''
    BadTeeths = ''    
   

    for element in data:  # iterate on each element of the list
    # element is a dict
        if element['ID'] == filename:  # get the id
            #print('ELEMENT LABEL',element['Label'].items())
            # for key, value in element['Label'].items():
            #     print(key, 'is the key for the value')
            #     print ('SWITCHER',switcher.get(key) )
            try:                   
            #         print ('SWITCHER',switcher.get(key) )
                try:
                    if element['Label']['Bad Teeth']:
                        BadTeeths = element['Label']['Bad Teeth']   
                        
                except:
                    print('error in BAD teeth')
                try:
                    if element['Label']['Bad Gums']:
                        BadGums = element['Label']['Bad Gums']                        
                except:
                    print('error in BAD gums')
                try:
                    if element['Label']['Good Gums']:
                        GoodGums = element['Label']['Good Gums']                        
                except:
                    print('error in GOOD gums')
                try:
                    if element['Label']['Good Teeth']:
                        GoodTeeths = element['Label']['Good Teeth']                        
                except:
                    print('error in GOOD teeths')
            except:
                print('Error photo{} has no labels'.format(element['ID']))
                noAnnotation = False
        else:
            noAnnotation = True
    if noAnnotation == False:
        print('Error photo{} has no annotations'.format(element['ID'])) 
        return
    #get diseases/good coordinates
    diseasesXY = []
    
    #Bad gums
    for badgum in BadGums:
        try:
            a = (badgum['select_disease_a'][0],badgum['geometry'])    
            #we change each disease by a badteeth label
            bmf = list(a)
            bmf[0] ='badgum'
            a = tuple(bmf)            
            diseasesXY.append(a)        
            # if 'select_disease_b' in badgum:
            #     a = (badgum['select_disease_b'][0],badgum['geometry'])
            #     diseasesXY.append(a)

            # if 'select_disease_c' in badgum:            
            #     a = (badgum['select_disease_c'][0],badgum['geometry'])
            #     diseasesXY.append(a)
        except:
            print ("Two diseases in one BadGum!")
    #Bad teeth         
    for badteeth in BadTeeths:
        try:
            a = (badteeth['select_disease_a'][0],badteeth['geometry'])            
            bmf = list(a)
            bmf[0] ='badteeth'
            a = tuple(bmf)            
            diseasesXY.append(a)
            
            # if 'select_disease_b' in badteeth:
            #     a = (badteeth['select_disease_b'][0],badteeth['geometry'])
            #     diseasesXY.append(a)


            # if 'select_disease_c' in badteeth:
            #     a = (badteeth['select_disease_c'][0],badteeth['geometry'])
            #     diseasesXY.append(a)
        except:
            print ("Two diseases in one Badteeth!")

    for goodgum in GoodGums:    
        a = ('goodgum',goodgum['geometry'])
        diseasesXY.append(a)

    for goodteeth in GoodTeeths:     
        a = ('goodteeth',goodteeth['geometry'])
        diseasesXY.append(a)   
    
    #print(diseasesXY[0])    
    for label in diseasesXY:
        #print(label[0])
        labels_in_dataset.append(label[0])        
    return(diseasesXY)

def get_class_id_4_classes(label):

    mappings = {        
        'goodteeth' : 1,
        'goodgum' : 2,
        'badteeth' : 3,
        'badgum' : 4,        
    }
    return (mappings[label])

def create_tf_example(root,image_file):
  # Check if annotation xml file exists
  filename, file_extension = os.path.splitext(image_file)

  with tf.gfile.GFile(os.path.join(root, image_file), 'rb') as fid:
     encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)

  if image.format != 'JPEG':
    raise ValueError('Image format not JPEG')


  with tf.gfile.GFile(os.path.join(root, image_file), 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  width, height = image.size
  print(image_file,width,height)
  BB = get_cordinates(filename)
  xmin = []
  ymin = []
  ymax = []
  xmax = []
  classes_id = []
  classes_text = []

  
  #for each label in all labels
  # BB = all labels
  # item = one label (4 coordinates, 1 class id, 1 class name)  
  
  if BB:
    
    for item in BB:
        coordinates = item[1]                
        Y=[]
        X=[]
        for coord in coordinates:
            Y.append(coord['y'])
            X.append(coord['x'])
            
        xmin.append(float(min(X)/width))
        xmax.append(float(max(X)/width))
        ymin.append(float(min(Y)/height))
        ymax.append(float(max(Y)/height)) 
#        classes_id.append(get_class_id(item[0]))
        classes_id.append(get_class_id_4_classes(item[0]))
        #print (item[0])
        classes_text.append(item[0].encode('utf8'))  
          
    print ('CLASSES_TEXT_after',classes_text)      
    enter = True
    if enter:   
        
        filename = image_file
        
    tf_example=[]
        
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
        #'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value = [bytes(image_file, encoding= 'utf-8')])),
        'image/source_id': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),        
        'image/object/class/label': dataset_util.int64_list_feature(classes_id),
    }))
    return tf_example

def train_files():
    trainFile = "/home/juanluis/workingrrr/smiletronix/models/research/VOCdevkit/VOC2012/ImageSets/Main/train.txt"
    trainXMLFiles = list()
    with open(trainFile, "rb") as f:
	    for line in f:
		    fileName = line.strip()
		    print (fileName)
		    trainXMLFiles.append(fileName + ".xml")
    rootDir = "/home/juanluis/workingrrr/smiletronix/models/research/VOCdevkit/VOC2012/Annotations"
    generateVOC2Json(rootDir, trainXMLFiles)


def main(_):
  writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
  images=[]  
  extensions = ['.jpeg','.jpg']      
  
  for root, dirs, files in os.walk(rootDir):
      for name in files:
        filename, file_extension = os.path.splitext(name)
        if file_extension in extensions:          
            images.append(name)   
        
  for image in images:    
    tf_example = create_tf_example(rootDir,image)     
    if tf_example:     
        writer.write(tf_example.SerializeToString())            
  writer.close()
   
  print ('labels in dataset',Counter(labels_in_dataset))
  print ('images in dataset',len(images))

if __name__ == '__main__':
  tf.app.run()