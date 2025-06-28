from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Create the ImageDataGenerator with basic rescaling
train_gen = ImageDataGenerator(rescale=1./255)

# Load images from dataset/train and prepare batches
train_flow = train_gen.flow_from_directory(
    'dataset/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Print the class labels detected from directory names
print("✅ Class Indices found in 'dataset/train':")
print(train_flow.class_indices)

