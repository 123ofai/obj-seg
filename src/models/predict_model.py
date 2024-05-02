import tensorflow as tf

def predict_model(model, image, gt_mask):
    pred_mask = model.predict(image)
    pred_mask = tf.math.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    print(image.shape, gt_mask.shape, pred_mask.shape)
    return pred_mask