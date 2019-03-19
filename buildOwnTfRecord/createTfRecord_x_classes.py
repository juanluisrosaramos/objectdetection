import tensorflow as tf
import os
import io
import json
from object_detection.utils import dataset_util
from collections import defaultdict
from PIL import Image
import PIL.Image
from collections import Counter
import cv2 
import imutils
import numpy as np

rootDir = 'test'
tfRecordName = 'tfrecords/test_p3p2_x_classes.tfrecord'
 
flags = tf.app.flags
flags.DEFINE_string('output_path', tfRecordName, 'Path to output TFRecord')
FLAGS = flags.FLAGS
json_file = 'project3.json'
with open(json_file, "r") as read_file:
     data = json.load(read_file)
labels_in_dataset = []
sizes = []


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
            #TODO avoid errors when JSON value is not found
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
                print('Error  annotation has{} no labels'.format(element['ID']))
                noAnnotation = False
        else:
            noAnnotation = True
    if noAnnotation == False:
        print('Error photo{} has no annotations'.format(element['ID'])) 
        return
    
    diseasesXY = []
    #TODO
    #Implement that One teeth/gum can have three diseases
    #Bad gums
    for badgum in BadGums:
        try:
            a = (badgum['select_disease_a'][0],badgum['geometry'])
            #print ('BAD_GUM',a[0],a[1])
            if a[0] in ['periodontal_disease','gum_recession']:                
                diseasesXY.append(a)
        except:
            print ("Two diseases in one BadGum!")
    #Bad teeth         
    for badteeth in BadTeeths:
        try:
            a = (badteeth['select_disease_a'][0],badteeth['geometry'])
            #print ('BAD_TEETH',a[0],a[1])
            if a[0] in ['plaque_buildup','erosion/abrasion/attrition/abfraction','caries','hypomineralization','discoloration/staining']:                
                diseasesXY.append(a)           
        except:
            print ("Two diseases in one Badteeth!")

    for goodgum in GoodGums:    
        a = ('goodgum',goodgum['geometry'])
        diseasesXY.append(a)

    for goodteeth in GoodTeeths:     
        a = ('goodteeth',goodteeth['geometry'])
        diseasesXY.append(a)   
    
    for label in diseasesXY:
        #print(label[0])
        labels_in_dataset.append(label[0])        
    return(diseasesXY)

def get_class_id(label):

    mappings = {        
        'goodteeth' : 1,
        'goodgum' : 2,
        'caries' : 3,
        'plaque_buildup' : 4,
        'hypomineralization' : 5,
        'discoloration/staining' : 6,
        'gum_recession' : 7,
        'periodontal_disease' : 8,                
        'erosion/abrasion/attrition/abfraction' : 9
    }
    return (mappings[label])
      
def drawBox(boxes, image):
    for i in range(0, len(boxes)):
        # changed color and width to make it visible
        # drawBox([[1, 0, float(xmin[0]), float(ymin[0]), float(xmax[0]), float(ymax[0])]], image)
        cv2.rectangle(image, (boxes[i][2], boxes[i][3]), (boxes[i][4], boxes[i][5]), (255, 0, 0), 1)
    cv2.imshow("img", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def create_tf_example(root,image_file):
  
  filename, file_extension = os.path.splitext(image_file)
  
  image = cv2.imread(os.path.join(root, image_file),3)
  height, width, channels = image.shape
#   cv2.imshow('dst_rt', image)
#   cv2.waitKey(0)
#   cv2.destroyAllWindows()
# In case of resizing of the image we have to rescale the coordinates of the BB
# we initialize to 1 
  scale_x = 1
  scale_y = 1
  if width > 1200:
    #coordinates
    #print ('Image: ',filename,' width ',width, ' resized')
    image = imutils.resize(image, width=1200)
    scale_x = image.shape[1] / width
    scale_y = image.shape[0] / height
    print('RESCALING y ',image.shape[0],'/ por' , height)

    height, width,channels = image.shape
    # print ('New size ',width,',',height)
  encoded_jpg = cv2.imencode('.jpg',image)[1].tostring()
  sizes.append(tuple(image.shape)) 

  BB = get_cordinates(filename)
  xmin = []
  ymin = []
  ymax = []
  xmax = []
  classes_id = []
  classes_text = []

  #xmin.text = str(np.round(int(xmin.text) * scale_x))

  #for each label in all labels
  # BB = all labels
  # item = one label (4 coordinates, 1 class id, 1 class name)    
  print('SCALE_X',scale_x,' SCALE_Y',scale_y)
  if BB:
    
    for item in BB:
        coordinates = item[1]                
        Y=[]
        X=[]
        # printmaxX=[]
        # printmaxY=[]
        # printminX=[]
        # printminY=[]
        for coord in coordinates:
            Y.append(coord['y'])
            X.append(coord['x'])
            
        xmin.append(float(np.round(min(X) * scale_x)/width))
        # print('MIN X scaled ',float(np.round(min(X) * scale_x)))
        # print('MIN X NO scaled ',float(np.round(min(X) * 1)))
        xmax.append(float(np.round(max(X) * scale_x)/width))
        ymin.append(float(np.round(min(Y) * scale_y)/height))
        ymax.append(float(np.round(max(Y) * scale_y)/height))

        # printminX.append(float(np.round(min(X) * scale_x)))
        # printmaxX.append(float(np.round(max(X) * scale_x)))
        # printminY.append(float(np.round(min(Y) * scale_y)))
        # printmaxY.append(float(np.round(max(Y) * scale_y)))
        classes_id.append(get_class_id(item[0]))
        #print ('CLASS_NAME:',item[0],' CLASS_ID:',get_class_id(item[0]))
        classes_text.append(item[0].encode('utf8'))  
        #print('xmin',float(xmin[0]),'xmax',xmax,'ymin',ymin,'ymax',ymax)      
       # drawBox([[1, 0, int(printminX[0]), int(printminY[0]), int(printmaxX[0]), int(printmaxY[0])]],np.array(image))

    enter = True
    if enter:         
        filename = image_file
        
    tf_example=[]
        
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
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
  av_width = sum(v[1] for v in list(sizes)) / float(len(sizes))
  av_height = sum(v[0] for v in list(sizes)) / float(len(sizes))
  print ('Average width:',av_width, 'Average height:',av_height)

if __name__ == '__main__':
  tf.app.run()