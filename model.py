import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

# Load preprocessed data
train_images = np.load('processed_dataset/train_images.npy')
train_labels = np.load('processed_dataset/train_labels.npy')
test_images = np.load('processed_dataset/test_images.npy')
test_labels = np.load('processed_dataset/test_labels.npy')

# Create a label encoder object
label_encoder = LabelEncoder()

# Fit and transform the labels to integers
train_labels = label_encoder.fit_transform(train_labels)
test_labels = label_encoder.transform(test_labels)

# Print the classes to see the mapping
print("Label Mapping:", label_encoder.classes_)

# Reshape the images to add channel dimension
train_images = train_images.reshape((-1, 48, 48, 1))
test_images = test_images.reshape((-1, 48, 48, 1))

# Define the CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(48, 48, 1)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(7, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

# Correct way to save the model according to the new Keras API
model.save('emotion_recognition_model.h5', save_format='h5')
