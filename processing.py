import os
import cv2
import numpy as np

def load_and_preprocess_images(directory, size=(48, 48)):
    images = []
    labels = []
    for foldername in os.listdir(directory):
        folder_path = os.path.join(directory, foldername)
        for filename in os.listdir(folder_path):
            filepath = os.path.join(folder_path, filename)
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, size)
                images.append(img)
                labels.append(foldername)  # Use the folder name as the label
    return np.array(images), np.array(labels)

def main():
    # Paths to your folders
    train_dir = 'dataset/train'
    test_dir = 'dataset/test'
    save_dir = 'processed_dataset'

    # Load and preprocess the images
    train_images, train_labels = load_and_preprocess_images(train_dir)
    test_images, test_labels = load_and_preprocess_images(test_dir)

    # Normalize the images
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # Save the processed images and labels
    np.save(os.path.join(save_dir, 'train_images.npy'), train_images)
    np.save(os.path.join(save_dir, 'train_labels.npy'), train_labels)
    np.save(os.path.join(save_dir, 'test_images.npy'), test_images)
    np.save(os.path.join(save_dir, 'test_labels.npy'), test_labels)

if __name__ == '__main__':
    main()
