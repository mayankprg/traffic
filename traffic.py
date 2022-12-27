import cv2 as cv
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # # Split data into training and testing sets
    # labels = tf.keras.utils.to_categorical(labels)
    # x_train, x_test, y_train, y_test = train_test_split(
    #     np.array(images), np.array(labels), test_size=TEST_SIZE
    # )

    # # Get a compiled neural network
    # model = get_model()

    # # Fit model on training data
    # model.fit(x_train, y_train, epochs=EPOCHS)

    # # Evaluate neural network performance
    # model.evaluate(x_test,  y_test, verbose=2)

    # # Save model to file
    # if len(sys.argv) == 3:
    #     filename = sys.argv[2]
    #     model.save(filename)
    #     print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """

    images = []
    labels = []

    for label in os.listdir(data_dir):
        labels.append(int(label))

        # path to nth label
        path = change_path(os.path.join(data_dir, label))

        for file in os.listdir(path):
            # path to the images
            img_path = change_path(os.path.join(data_dir, label, file))

            # read image
            image = cv.imread(f'{img_path}', -1)

            # if height or width not equal to 30 then resize
            height, width = image.shape[:2]
            if not height == IMG_HEIGHT or not width == IMG_WIDTH:

                # resize the image to IMG_HEIGHT, IMG_WIDTH
                image = cv.resize(image, (IMG_HEIGHT, IMG_WIDTH),
                                  interpolation=cv.INTER_AREA)

            # save image in images array
            images.append(image)

    # return tuple of images and labels
    return (images, labels)


def change_path(path):
    if sys.platform == "win32":
        return path
    else:
        return path.replace('\\', "/")


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
