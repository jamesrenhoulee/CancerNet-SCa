# Training, Evaluation and Inference
CancerNet-SCa models takes as input an image of shape (N, 224, 224, 3) and outputs the softmax probabilities as (N, 2), where N is the number of batches.
If using the TF checkpoints, here are some useful tensors:

* input tensor: `input_1:0`
* logit tensor: `probs/MatMul:0`
* output tensor: `probs/Softmax:0`
* label tensor: `probs_target:0`
* loss tensor: `loss/mul:0`
* training placeholder tensor: `keras_learning_phase:0` 

## Steps for training
TF training script from a pretrained model:
1. We provide you with the tensorflow training script, [train_tf.py](../train_tf.py)
2. Locate the tensorflow checkpoint files (location of pretrained model)
3. To train from a pretrained model:
```
python train_tf.py \
    --weightspath models/CancerNet-SCa-A \
    --metaname model.meta \
    --ckptname model-0 \
    --trainfile train_images.csv \
    --testfile val_images.csv \
```
4. For more options and information, `python train_tf.py --help`

## Steps for evaluation

1. We provide you with the tensorflow evaluation script, [eval.py](../eval.py)
2. Locate the tensorflow checkpoint files
3. To evaluate a tf checkpoint:
```
python eval.py \
    --weightspath models/CancerNet-SCa-A \
    --metaname model.meta \
    --ckptname model-0
```
4. For more options and information, `python eval.py --help`

## Steps for inference
**DISCLAIMER: Do not use this prediction for self-diagnosis. You should check with your local authorities for the latest advice on seeking medical assistance.**

1. Download a model from the [pretrained models section](models.md)
2. Locate models and image to be inferenced
3. To inference,
```
python inference.py \
    --weightspath models/CancerNet-SCa-A \
    --metaname model.meta \
    --ckptname model-0 \
    --imagepath assets/predict_this.jpeg
```
4. For more options and information, `python inference.py --help`