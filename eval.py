from sklearn.metrics import confusion_matrix
import numpy as np
import os, argparse
import cv2

import tensorflow as tf
# To remove TF Warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from data import ISIC_Dataset


def eval(sess, graph, eval_iters, input_tensor, output_tensor, training_placeholder, input_size):
    image_tensor = graph.get_tensor_by_name(input_tensor)
    pred_tensor = graph.get_tensor_by_name(output_tensor)
    train_placeholder = graph.get_tensor_by_name(training_placeholder)

    y_test = []
    pred = []
    for i in range(eval_iters):
        data_feeds = sess.run(test_inputs)
        y_test.append(data_feeds['label/one_hot'].argmax(axis=-1))
        pred.append(np.array(sess.run(pred_tensor, feed_dict={image_tensor: data_feeds['image'], train_placeholder: 0})).argmax(axis=1))
    y_test = np.array(y_test)
    pred = np.array(pred)

    matrix = confusion_matrix(y_test, pred)
    matrix = matrix.astype('float')
    print(matrix)
    class_acc = [matrix[i,i]/np.sum(matrix[i,:]) if np.sum(matrix[i,:]) else 0 for i in range(len(matrix))]
    print('Sensitivity Benign: {0:.3f}, Malignant: {1:.3f}'.format(class_acc[0], class_acc[1]))
    ppvs = [matrix[i,i]/np.sum(matrix[:,i]) if np.sum(matrix[:,i]) else 0 for i in range(len(matrix))]
    print('PPV Benign: {0:.3f}, Malignant {1:.3f}'.format(ppvs[0], ppvs[1]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='COVID-Net Evaluation')
    parser.add_argument('--weightspath', default='models/CancerNet-SCa-A', type=str, help='Path to output folder')
    parser.add_argument('--metaname', default='model.meta', type=str, help='Name of ckpt meta file')
    parser.add_argument('--ckptname', default='model-0', type=str, help='Name of model ckpts')
    parser.add_argument('--testfile', default='test_images.csv', type=str, help='Name of testfile')
    parser.add_argument('--testfolder', default='data/test/', type=str, help='Folder where test data is located')
    parser.add_argument('--in_tensorname', default='input_1:0', type=str, help='Name of input tensor to graph')
    parser.add_argument('--out_tensorname', default='probs/Softmax:0', type=str, help='Name of output tensor from graph')
    parser.add_argument('--training_placeholder', default='keras_learning_phase:0', type=str, help='Name of training placeholder tensor from graph')
    parser.add_argument('--input_size', default=224, type=int, help='Size of input (ex: if 224x224, --input_size 224)')

    args = parser.parse_args()

    # Create TF test dataset and make it iterable for evaluation
    isic_dataset = ISIC_Dataset(args.testfolder, args.testfile, is_training=False, batch_size=1,
         input_shape=(args.input_size, args.input_size), balance_dataset=False)

    test_dataset = isic_dataset.create_tf_dataset()
    test_dataset_iter = test_dataset.make_initializable_iterator()
    test_inputs = test_dataset_iter.get_next()
    num_test_samples = isic_dataset.len_data
    
    print('was able to make the dataset')

    sess = tf.Session()
    sess.run([test_dataset_iter.initializer, tf.tables_initializer()])
    tf.get_default_graph()
    saver = tf.train.import_meta_graph(os.path.join(args.weightspath, args.metaname))
    saver.restore(sess, os.path.join(args.weightspath, args.ckptname))

    graph = tf.get_default_graph()

    eval(sess, graph, num_test_samples, args.in_tensorname, args.out_tensorname, args.training_placeholder, args.input_size)