from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# Data paths
train_dir = 'dataset/train'
val_dir = 'dataset/val'

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest'
)

# No augmentation for validation, just rescaling
val_datagen = ImageDataGenerator(rescale=1./255)

# Data generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Load pre-trained VGG16 model without the top layer
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# Build the model
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dense(train_generator.num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Save best model
checkpoint = ModelCheckpoint("model/vgg16_augmented_model.h5", monitor='val_accuracy', save_best_only=True)

# Train the model
model.fit(
    train_generator,
    epochs=25,
    validation_data=val_generator,
    callbacks=[checkpoint]
)
# Save the trained model
model.save("butterfly_model_v1.h5")
print("✅ Model saved as butterfly_model_v1.h5")

# Save class indices to JSON file
import json

with open("class_indices.json", "w") as f:
    json.dump(train_flow.class_indices, f)

print("✅ Class indices saved to class_indices.json")

