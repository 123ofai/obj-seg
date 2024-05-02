import sys
sys.path.append('../src')

import tensorflow as tf
from data.make_dataset import create_dataset
from models.unet_model import define_model
from visualization.visualize import visualise_test
import datetime

def train():
    ds_train, ds_test = create_dataset()
    model = define_model()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    # Save tensorboard
    log_dir = "logs/exp1/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Model training
    history = model.fit(ds_train,
                        epochs=10,
                        validation_data=ds_test,
                        callbacks=[tensorboard_callback])
    
    # Save Model
    

    # Save test images
    visualise_test(model, ds_test)
    


