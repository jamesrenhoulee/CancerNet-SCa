import numpy as np
import tensorflow as tf
import os, argparse
import cv2

from data import process_image_file

parser = argparse.ArgumentParser(description='CancerNet-SCa Inference')
parser.add_argument('--weightspath', default='models/CancerNet-SCa', type=str, help='Path to output folder')
parser.add_argument('--metaname', default='model.meta', type=str, help='Name of ckpt meta file')
parser.add_argument('--ckptname', default='model-0', type=str, help='Name of model ckpts')
parser.add_argument('--imagepath', default='assets/predict_this.jpeg', type=str, help='Full path to image to be inferenced')
parser.add_argument('--in_tensorname', default='input_1:0', type=str, help='Name of input tensor to graph')
parser.add_argument('--out_tensorname', default='probs/Softmax:0', type=str, help='Name of output tensor from graph')
parser.add_argument('--input_size', default=224, type=int, help='Size of input (ex: if 224x224, --input_size 224)')

args = parser.parse_args()

mapping = {'benign': 0, 'malignant': 1}
inv_mapping = {0: 'benign', 1: 'malignant'}

sess = tf.Session()
tf.get_default_graph()
saver = tf.train.import_meta_graph(os.path.join(args.weightspath, args.metaname))
saver.restore(sess, os.path.join(args.weightspath, args.ckptname))

graph = tf.get_default_graph()

image_tensor = graph.get_tensor_by_name(args.in_tensorname)
pred_tensor = graph.get_tensor_by_name(args.out_tensorname)

x = process_image_file(args.imagepath, args.input_size)
x = x.astype('float32') / 255.0
pred = sess.run(pred_tensor, feed_dict={image_tensor: np.expand_dims(x, axis=0)})

print('Prediction: {}'.format(inv_mapping[pred.argmax(axis=1)[0]]))
print('Confidence')
print('Benign: {:.3f}, Malignant: {:.3f}'.format(pred[0][0], pred[0][1]))
print('**DISCLAIMER**')
print('Do not use this prediction for self-diagnosis. You should check with your local authorities for the latest advice on seeking medical assistance.')