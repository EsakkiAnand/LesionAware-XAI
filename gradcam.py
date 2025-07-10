import tensorflow as tf
from tensorflow.keras.models import Model
import numpy as np
import cv2

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, classifier_layer_name, pred_index=None):
    grad_model = Model(inputs=[model.inputs], outputs=[
        model.get_layer(last_conv_layer_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy(), int(pred_index)

def binarize_heatmap(heatmap, threshold=0.5):
    return (heatmap > threshold).astype(np.uint8)