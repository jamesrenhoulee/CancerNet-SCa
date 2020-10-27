"""
Generates the necessary TensorFlow model files from a Keras model to create a model entity
in GenSynth.
The script also displays the input output, and training placeholder tensors of the model
to be inputted when creating the GenSynth model entity. 
"""
import sys
import argparse

from model_tools.utils import *
from model_tools.visualization import *

def print_tensor_details(tensor_list):
    for tensor in tensor_list:
        print("Tensor name: '{}', shape: {}, datatype: {}".format(
            tensor.name, tensor.shape, tensor.dtype.name
        ))

def main(argv):
    parser = argparse.ArgumentParser(description='Keras model entity tool for Tensorflow conversion.')
    parser.add_argument('--h5-model-file', type=str, help='Saved Keras model file in HDF5 format.')
    parser.add_argument('--output-directory', type=str, default='./saved_model',
                        help='Path to save output model META and checkpoint files.')
    args = parser.parse_args(argv[1:])

    # Clear backend keras session
    clear_sess()

    # Load keras model from h5 file
    keras_model = get_keras_model(args.h5_model_file)

    # Get input tensors for keras model and some description (name, size, etc.)
    input_tensors = get_input_tensors(keras_model)

    # Get output tensors for keras model
    # Note: to get the loss tensor, set Model outputs to point to loss layer
    output_tensors = get_output_tensors(keras_model)

    # Get tf.Session from keras backend session
    sess = get_sess()

    # Get tf.Graph from keras model
    graph = get_graph(sess=sess)

    # Initialize graph collections based on keras model attributes
    init_keras_collections(graph, keras_model)

    # Get training placeholder tensor
    training_placeholder_tensor = get_training_placeholder(graph)

    # Generate META and checkpoint files
    file_paths = save_tf_model(sess, args.output_directory)

    # Print model details to be inputted into GenSynth
    print('========== Model Files ==========')
    print("Save Type: Meta & Checkpoint")
    print("Meta File Location: {}".format(file_paths['meta_file_path']))
    print("Checkpoint Folder: {}".format(file_paths['checkpoint_dir']))
    print()
    print('========== Graph Information ==========')
    print('Input Tensors:')
    print_tensor_details(input_tensors)
    print()
    print('Output Tensors:')
    print_tensor_details(output_tensors)
    print()
    if training_placeholder_tensor != None:
        print('Training Placeholder Tensor:')
        print_tensor_details([training_placeholder_tensor])


if __name__ == "__main__":
    main(sys.argv)
