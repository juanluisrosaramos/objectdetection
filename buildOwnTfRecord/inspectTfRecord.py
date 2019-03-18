import tensorflow as tf
from google.protobuf.json_format import MessageToJson
import numpy as np


for example in tf.python_io.tf_record_iterator("x_classes/test_x_classes.tfrecord"):
    result = tf.train.Example.FromString(example)
    jsonMessage = MessageToJson(tf.train.Example.FromString(example))

for serialized_example in tf.python_io.tf_record_iterator('x_classes/test_x_classes.tfrecord'):
    example = tf.train.Example()
    example.ParseFromString(serialized_example)
    print(example.features.feature['image/filename'].bytes_list.value)
    print(np.array(example.features.feature['int'].int64_list.value))
    print(np.array(example.features.feature['flo'].float_list.value))

#print(jsonMessage)