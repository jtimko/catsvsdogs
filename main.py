import tensorflow as tf
import numpy as np
import os

# this limits the error messages you may see when running on metal
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# setting up directories
dir_path = '/path/to/training/set'
cats_dir = os.path.join(dir_path, 'cats')
dogs_dir = os.path.join(dir_path, 'dogs')

# path to picture to test
test_img = '/path/to/testing/photo.jpg'

train_ds = tf.keras.utils.image_dataset_from_directory(
    dir_path,
    validation_split=0.2,
    subset="training",
    image_size=(180, 180),
    batch_size=32,
    seed=42
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    dir_path,
    validation_split=0.2,
    subset="validation",
    image_size=(180, 180),
    batch_size=32,
    seed=42
)

# caching and prefetching
class_names = train_ds.class_names
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
val_ds = train_ds.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

data_augumentaion = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=42),
    tf.keras.layers.RandomRotation(0.1, seed=42)
])

NUM_CLASSES = len(class_names)

# creating the layers ... Neural Network
model = tf.keras.Sequential(
    [
        tf.keras.layers.Rescaling(1./255, input_shape=(180, 180, 3)),
        data_augumentaion,
        tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(NUM_CLASSES)
    ]
)

model.compile(
    optimizer="Adam",
    loss= tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# training
history = model.fit(train_ds, validation_data=val_ds, epochs=15)

# the comparison of your photo and the model
img = tf.keras.utils.load_img(test_img, target_size=(180,180))

img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print("This picture is a {} with {:.2f} confidence.".format(class_names[np.argmax(score)], 100 * np.max(score)))