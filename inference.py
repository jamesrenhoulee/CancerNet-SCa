import numpy as np
import tensorflow as tf
import os, argparse
import cv2

# To remove TF Warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def process_image_file(sess, filename, input_size):
    img_decoded = tf.image.decode_jpeg(tf.io.read_file(filename), channels=3)
    img = tf.image.resize_images(img_decoded, (input_size, input_size))
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)/255.0
    image = sess.run(img)    
    return image


parser = argparse.ArgumentParser(description='CancerNet-SCa Inference')
parser.add_argument('--weightspath', default='models/CancerNet-SCa-A', type=str, help='Path to output folder')
parser.add_argument('--metaname', default='model.meta', type=str, help='Name of ckpt meta file')
parser.add_argument('--ckptname', default='model-0', type=str, help='Name of model ckpts')
parser.add_argument('--imagepath', default='assets/ex_malignant.jpg', type=str, help='Full path to image to be inferenced')
parser.add_argument('--in_tensorname', default='input_1:0', type=str, help='Name of input tensor to graph')
parser.add_argument('--out_tensorname', default='probs/Softmax:0', type=str, help='Name of output tensor from graph')
parser.add_argument('--training_placeholder', default='keras_learning_phase:0', type=str, help='Name of training placeholder tensor from graph')
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
train_placeholder = graph.get_tensor_by_name(args.training_placeholder)

x = process_image_file(sess, args.imagepath, args.input_size)
pred = sess.run(pred_tensor, feed_dict={image_tensor: np.expand_dims(x, axis=0), train_placeholder: 0})

print('Prediction: {}'.format(inv_mapping[pred.argmax(axis=1)[0]]))
print('Confidence')
print('Benign: {:.3f}, Malignant: {:.3f}'.format(pred[0][0], pred[0][1]))
print('**DISCLAIMER**')
print('Do not use this prediction for self-diagnosis. You should check with your local authorities for the latest advice on seeking medical assistance.')