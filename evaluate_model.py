from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Load your model
model = load_model('model/vgg16_butterfly_model.h5')

# Test data directory (create if you don't have one)
test_dir = 'test_data'  # replace this with your actual test images folder path

# Image generator for test images
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical'
)

# Evaluate model on test data
loss, acc = model.evaluate(test_generator)
print(f"Test Accuracy: {acc * 100:.2f}%")

