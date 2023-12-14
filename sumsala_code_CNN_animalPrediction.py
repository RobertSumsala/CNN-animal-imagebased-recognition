# ZADANIE 03 - Robert Sumsala
import math
import os
import pathlib
import pickle

import PIL
import numpy as np
from keras import regularizers
from keras.src.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
import pandas as pd
import tensorflow as tf
import PIL.Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import confusion_matrix
from tensorflow.python.keras import models
import seaborn as sns
from tensorflow.python.keras.callbacks import ModelCheckpoint

print()
print("- - - - - - - ZADANIE 03 - - - - - - -")
print()
print()

# Allow printing more columns
pd.options.display.width = None
pd.options.display.max_columns = None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)

# Do not show pandas warnings
pd.set_option('mode.chained_assignment', None)


# ----------------------- EXTRA FUNCTIONS ----------------------------------------------------------------------------------------------------------

def create_animal_collage(train_dir, images_per_row=10):
    animals_folders = list(pathlib.Path(train_dir).glob('*'))

    images = []
    for animal_folder in animals_folders:
        animal_images = list(animal_folder.glob('*'))
        print(f"Class: {animal_folder.stem}, Number of Pictures: {len(animal_images)}")
        if animal_images:
            im = PIL.Image.open(str(animal_images[0]))
            im = im.resize((128, 128))
            images.append(im)

    # Calculate the number of rows needed
    num_rows = math.ceil(len(images) / images_per_row)

    # Calculate the size of the collage
    collage_width = max(im.width for im in images) * images_per_row
    collage_height = max(im.height for im in images) * num_rows

    # Create a blank canvas for the collage
    collage = PIL.Image.new('RGB', (collage_width, collage_height))

    # Paste each image onto the canvas
    x_offset, y_offset = 0, 0
    for im in images:
        collage.paste(im, (x_offset, y_offset))
        x_offset += im.width
        if x_offset >= collage_width:
            x_offset = 0
            y_offset += im.height

    # Display the collage
    collage.show()


def get_x_and_y(dataset):
    # Extracting data and labels from the training dataset
    X = []
    y = []

    # Initialize a set to keep track of unique class labels
    unique_labels = set()

    for images, labels in dataset:
        X.append(images.numpy())
        y.append(labels.numpy())

        # Update the set of unique class labels
        unique_labels.update(labels.numpy())

    X = tf.concat(X, axis=0)
    y = tf.concat(y, axis=0)

    return X, y, unique_labels


def preprocess_images(image_paths):
    images = [tf.keras.preprocessing.image.load_img(img_path, target_size=(img_height, img_width)) for img_path in
              image_paths]
    images = [tf.keras.preprocessing.image.img_to_array(img) for img in images]
    images = [tf.keras.applications.resnet50.preprocess_input(img) for img in images]
    return np.stack(images)


def get_predictions_and_accuracy(animal_name, image_paths):
    # Preprocess images
    X_test_category = preprocess_images(image_paths)

    # Get predictions
    preds = model_resnet50.predict(X_test_category)
    decoded_preds = tf.keras.applications.resnet50.decode_predictions(preds, top=1)

    top_prediction = decoded_preds[0]
    imagenet_id, label, score = top_prediction[0]

    correctly_predicted = 0
    for prediction in decoded_preds:
        imagenet_id, label, score = prediction[0]
        if label == animal_name:
            correctly_predicted += 1
    print(f"Top prediction byt ImageNet: {label} -> {score:.2f}")
    print(f"Animal name in the dataset: {animal_name}")
    print(f"Correctly predicted {correctly_predicted}/6")


def get_predictions_train(model, dataset):
    y_pred = []  # store predicted labels
    y_true = []  # store true labels

    # iterate over the dataset
    for image_batch, label_batch in dataset:
        # append true labels
        y_true.append(label_batch)
        # compute predictions
        preds = model.predict(image_batch, verbose=0)
        # append predicted labels
        y_pred.append(np.argmax(preds, axis=- 1))

    # convert the true and predicted labels into tensors
    y_true = tf.concat([item for item in y_true], axis=0)
    y_pred_classes = tf.concat([item for item in y_pred], axis=0)
    class_labels = list(dataset.class_names)

    return y_true, y_pred_classes, class_labels


def get_predictions_test(model, dataset):
    # get predictions from test data
    y_pred = model.predict(dataset)
    # convert predictions classes to one hot vectors
    y_pred_classes = np.argmax(y_pred, axis=1)
    # get true labels for test set
    y_true = np.concatenate([y for x, y in dataset], axis=0)
    class_labels = list(dataset.class_names)

    return y_true, y_pred_classes, class_labels


def plot_confusion_matrix(y_true, y_pred_classes, class_labels, dataset_name):
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred_classes)

    # Plot confusion matrix
    plt.figure(figsize=(64, 48))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.title(f'Confusion Matrix - {dataset_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


def plot_training_history(history):
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Test'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.tight_layout()
    plt.show()


def extract_labels_and_images_from_ds(dataset):
    # Extract all images and labels
    all_images = []
    all_labels = []

    for images, labels in train_dataset.as_numpy_iterator():
        all_images.append(images)
        all_labels.append(labels)

    # Convert the lists to numpy arrays
    all_images = np.concatenate(all_images, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return all_images, all_labels


def get_image_paths_and_labels(dir_path):
    animals_folders = list(pathlib.Path(dir_path).glob('*'))

    images = []
    labels = []

    for animal_folder in animals_folders:
        animal_name = animal_folder.name
        animal_images = list(animal_folder.glob('*'))

        # Add image paths and labels
        images.extend(["./" + str(image) for image in animal_images])
        labels.extend([animal_name] * len(animal_images))

    return images, labels


def create_feature_df():
    # Create an empty data list
    feature_columns = [f"feature_{i}" for i in range(512)]
    data = {'image_path': [], 'label': []}
    for col in feature_columns:
        data[col] = []

    image_paths, labels = get_image_paths_and_labels(train_data_path)

    for image_path in image_paths:
        label = labels[image_paths.index(image_path)]
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(img_height, img_width))
        x = tf.keras.preprocessing.image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = tf.keras.applications.vgg16.preprocess_input(x)

        features = base_model.predict(x)
        features = features.flatten()

        # Append data to lists
        data['image_path'].append(image_path)
        data['label'].append(label)
        for col, feature in zip(feature_columns, features):
            data[col].append(feature)

    dataframe = pd.DataFrame(data)
    return dataframe


def get_random_images_from_cluster(df, cluster_id, num_images=100):
    cluster_df = df[df['cluster'] == cluster_id]

    num_images = min(num_images, len(cluster_df))

    # Randomly sample num_images from the cluster
    random_images = cluster_df.sample(n=num_images, random_state=42)

    return random_images


def create_collage(image_paths, output_path):
    images_per_row = 10
    common_size = (200, 200)
    images = [PIL.Image.open(image_path).resize(common_size) for image_path in image_paths]

    # Calculate the total number of rows needed
    num_rows = len(images) // images_per_row + (len(images) % images_per_row > 0)

    collage_width = images_per_row * common_size[0]
    collage_height = num_rows * common_size[1]

    collage = PIL.Image.new('RGB', (collage_width, collage_height))
    x_offset, y_offset = 0, 0

    for i, image in enumerate(images):
        collage.paste(image, (x_offset, y_offset))
        x_offset += common_size[0]

        # Move to the next row after placing 5 images
        if (i + 1) % images_per_row == 0:
            x_offset = 0
            y_offset += common_size[1]

    collage.save(output_path)


def create_average_image(image_paths, output_path):
    common_size = (200, 200)
    images = [PIL.Image.open(image_path).resize(common_size) for image_path in image_paths]
    average_image = PIL.Image.blend(images[0], images[1], 0.3)

    for image in images[2:]:
        average_image = PIL.Image.blend(average_image, image, 0.3)

    average_image.save(output_path)


# ----------------------- PREPARING & ANALYSING THE DATASET ----------------------------------------------------------------------------------------------------------

train_data_path = './data/train'
test_data_path = './data/test'

batch_size = 32
img_height = 128
img_width = 128

train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_data_path,
    shuffle=True,
    validation_split=0.1,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
)

valid_dataset = tf.keras.utils.image_dataset_from_directory(
    train_data_path,
    shuffle=True,
    validation_split=0.1,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_data_path,
    shuffle=False,  # No need to shuffle the test dataset
    image_size=(img_height, img_width),
    batch_size=batch_size,
)

# Display one picture for each animal class and print number of pictures in each class
create_animal_collage(train_data_path)


# ----------------------- ANALYSING THE DATASET (BASED ON ImageNet) ----------------------------------------------------------------------------------------------------------


# Get list of image paths and corresponding labels
X_test_paths = []
y_test_labels = []

for category_name in os.listdir(test_data_path):
    category_path = os.path.join(test_data_path, category_name)
    if os.path.isdir(category_path):
        for file_name in os.listdir(category_path):
            if file_name.endswith('.jpg'):
                image_path = os.path.join(category_path, file_name)
                X_test_paths.append(image_path)
                y_test_labels.append(category_name)

unique_labels_test = np.unique(y_test_labels)

# Prepare the model pretrained on the ImageNet problem
model_resnet50 = tf.keras.applications.resnet50.ResNet50(include_top=True,
                                                         weights='./ImageNetWeights/official_resnet50_weights.h5',
                                                         input_tensor=None,
                                                         input_shape=(224, 224, 3),
                                                         pooling=None,
                                                         classes=1000,
                                                         classifier_activation='softmax',
                                                         )
category_num = 0
for category_name in unique_labels_test:
    category_image_paths = [path for path, label in zip(X_test_paths, y_test_labels) if label == category_name]

    # Print category name and get predictions
    print(category_num)
    get_predictions_and_accuracy(category_name, category_image_paths)
    category_num += 1


# ----------------------- TRAINING OUR OWN CONVOLUTIONAL NEURAL NETWORK ----------------------------------------------------------------------------------------------------------


layers = tf.keras.layers

augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
        layers.RandomZoom(0.2),
        layers.RandomRotation(0.2),
    ]
)

model = tf.keras.Sequential(
    [
        augmentation,
        layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(32, 3, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.4),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(90, name="outputs")
    ]
)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

# Compile the model
model.compile(optimizer=optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

train_images, train_labels = extract_labels_and_images_from_ds(train_dataset)
valid_images, valid_labels = extract_labels_and_images_from_ds(valid_dataset)
test_images, test_labels = extract_labels_and_images_from_ds(test_dataset)

history = model.fit(
    train_images,
    train_labels,
    epochs=20,
    validation_data=(valid_images, valid_labels)
)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_dataset)
print(f'Test accuracy: {test_acc:.2%}')

train_loss, train_accuracy = model.evaluate(train_dataset)
print(f'Final Train Accuracy: {train_accuracy:.2%}')

# Create graphs to evaluate this CNN
# Accuracy and Loss graphs
plot_training_history(history)

# Confusion matrices
y_true, y_pred_classes, class_labels = get_predictions_train(model, train_dataset)
plot_confusion_matrix(y_true=y_true, y_pred_classes=y_pred_classes, class_labels=class_labels, dataset_name='train')

y_true, y_pred_classes, class_labels = get_predictions_test(model, test_dataset)
plot_confusion_matrix(y_true=y_true, y_pred_classes=y_pred_classes, class_labels=class_labels, dataset_name='test')


# ----------------------- IMPROVING CNN USING TRANSFER LEARNING ----------------------------------------------------------------------------------------------------------


base_model = tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet')

# Create a dataframe of features for each image from the train folder
df = create_feature_df()

df.to_csv('./data/features_dataframe.csv', index=False)

# load saved dataframe
df = pd.read_csv('./data/features_dataframe.csv')

# Extract features for clustering
feature_columns_indices = list(range(2, len(df.columns)))
features = df.iloc[:, feature_columns_indices].to_numpy()


# Perform K-means clustering
num_clusters = 3
cluster = AgglomerativeClustering(n_clusters=num_clusters, linkage='ward')
df['cluster'] = cluster.fit_predict(features)

# Create directories to save collages and average images
os.makedirs('./data/collages', exist_ok=True)
os.makedirs('./data/average_images', exist_ok=True)

# Iterate over clusters
for cluster_id in range(num_clusters):
    random_cluster_df = get_random_images_from_cluster(df, cluster_id)

    # Create collage
    collage_path = f'./data/collages/collage_cluster_{cluster_id}.png'
    create_collage(random_cluster_df['image_path'].tolist(), collage_path)

    # Calculate average image
    average_image_path = f'./data/average_images/average_image_cluster_{cluster_id}.png'
    create_average_image(random_cluster_df['image_path'].tolist(), average_image_path)


AUTOTUNE = tf.data.AUTOTUNE

train_dataset_pref = train_dataset.prefetch(buffer_size=AUTOTUNE)
valid_dataset_pref = valid_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset_pref = test_dataset.prefetch(buffer_size=AUTOTUNE)

data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip('horizontal'),
  tf.keras.layers.RandomRotation(0.2),
])

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input


# Create the base pre-trained model
base_model = tf.keras.applications.MobileNetV2(input_shape=(img_height, img_width, 3),
                                               include_top=False,
                                               weights='imagenet')

image_batch, label_batch = next(iter(train_dataset_pref))
feature_batch = base_model(image_batch)

base_model.trainable = False

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)

prediction_layer = tf.keras.layers.Dense(90)
prediction_batch = prediction_layer(feature_batch_average)

inputs = tf.keras.Input(shape=(img_height, img_width, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_average_layer(x)
x = layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)


base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Fine-Tuning: Find how many layers of the base model we need to train and how many to freeze
# Train only on few epochs for now
history = model.fit(train_dataset_pref,
                    epochs=3,
                    validation_data=valid_dataset_pref)

model.save('./data/final_model.keras')

# Save the training history directly
with open('./data/training_history.pkl', 'wb') as file:
    pickle.dump(history.history, file)

# # Load model
# model = tf.keras.models.load_model('./data/final_model.keras')

# # Load the saved history
# with open('./data/training_history.pkl', 'rb') as file:
#     history = pickle.load(file)


# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_dataset_pref)
print(f'Test accuracy: {test_acc:.2%}')

train_loss, train_accuracy = model.evaluate(train_dataset_pref)
print(f'Final Train Accuracy: {train_accuracy:.2%}')

# Create graphs to evaluate this CNN
# Accuracy and Loss graphs
plot_training_history(history)

# Confusion matrices
y_true, y_pred_classes, class_labels = get_predictions_train(model, train_dataset)
plot_confusion_matrix(y_true=y_true, y_pred_classes=y_pred_classes, class_labels=class_labels, dataset_name='train')

y_true, y_pred_classes, class_labels = get_predictions_test(model, test_dataset)
plot_confusion_matrix(y_true=y_true, y_pred_classes=y_pred_classes, class_labels=class_labels, dataset_name='test')


# ----------------------- BONUS 1: TESTING OUR CNN ON OUR OWN PICTURES ----------------------------------------------------------------------------------------------------------

bonus_test_path = './data/bonus_test/catx'
class_names = train_dataset.class_names

for img_name in os.listdir(bonus_test_path):
    if not (img_name.endswith('.JPG') or img_name.endswith('.jpg')):
        continue
    img_path = os.path.join(bonus_test_path, img_name)

    # Load and preprocess the image
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(img_height, img_width))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)


    # Make predictions
    predictions = model.predict(img_array)
    
    # Get the predicted class index (assuming predictions is a numpy array)
    predicted_class_index = np.argmax(predictions)

    # Print the results
    print(f"Image: {img_name}")
    print(f"Predicted Class: {predicted_class_index}")
    print("-------------------")







print('SUCCESS')

# -----------------------------------------------------------------------------------------------------------------------------------------------

# RESOURCES
    # Image generator - Stackoverflow, seminar10
    # Display pictures as collage - ChatGPT
    # Help with using pretrained ImageNet - https://learnopencv.com/image-classification-pretrained-imagenet-models-tensorflow-keras/
    # Help with our CNN - tensorflow documentation
    # Confusion matrix - seminar11
    # Features extraction, Transfer learning/Fine-tuning - keras documentation
