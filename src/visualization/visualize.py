import matplotlib.pyplot as plt
import tensorflow as tf

def visualise_dataset(ds):
    # Displaying the dataset
    for example in ds:
        image = example["image"]
        mask = example["segmentation_mask"]

        plt.figure(figsize=(10, 10))
        plt.subplot(1, 5, 1)
        plt.title('Image')
        plt.imshow(image)
        plt.savefig('../../reports/outputs/image1.png')

        plt.subplot(1, 5, 2)
        plt.title('Mask')
        plt.imshow(mask)
        plt.savefig('../../reports/outputs/mask1.png')

        # Plotting the individual masks
        plt.subplot(1, 5, 3)
        plt.title('Mask=1')
        plt.imshow(mask==1)
        plt.savefig('../../reports/outputs/mask_idx1.png')

        plt.subplot(1, 5, 4)
        plt.title('Mask=2')
        plt.imshow(mask==2)
        plt.savefig('../../reports/outputs/mask_idx2.png')

        plt.subplot(1, 5, 5)
        plt.title('Mask=3')
        plt.imshow(mask==3)
        plt.savefig('../../reports/outputs/mask_idx3.png')
        break

def visualise_test(model, ds_test):
    for (image, mask) in ds_test:
        pred_mask = model.predict(image)
        pred_mask = tf.math.argmax(pred_mask, axis=-1)
        pred_mask = pred_mask[..., tf.newaxis]
        print(image.shape, mask.shape, pred_mask.shape)

        plt.figure(figsize=(10, 10))
        plt.subplot(1, 3, 1)
        plt.title('Image')
        plt.imshow(image[0])
        plt.savefig('../../reports/outputs/image.png')

        plt.subplot(1, 3, 2)
        plt.title('GT Mask')
        plt.imshow(mask[0])
        plt.savefig('../../reports/outputs/gt_mask.png')

        plt.subplot(1, 3, 3)
        plt.title('Pred Mask')
        plt.imshow(pred_mask[0])
        plt.savefig('../../reports/outputs/output.png')
        break
