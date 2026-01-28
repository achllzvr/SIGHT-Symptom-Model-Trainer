import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

# --- CONFIGURATION ---
DATASET_PATH = 'sight_dataset'  # Point this to your main folder
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20  # How many times to study the dataset

# 1. Prepare Data (with Augmentation to fake more data)
train_datagen = ImageDataGenerator(
    rescale=1./255,         # Normalize pixel values to 0-1
    rotation_range=20,      # Rotate images slightly
    horizontal_flip=True,   # Mirror images
    validation_split=0.2    # Use 20% of data for testing
)

train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Print the class labels so you know which ID is which disease
print("Class Indices:", train_generator.class_indices)

# 2. Load the "Professor" (MobileNetV2)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False # Freeze the base layers

# 3. Add Custom "Eye Doctor" Layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x) # Prevents memorizing
predictions = Dense(4, activation='softmax')(x) # 4 classes: Healthy, Uveitis, Cataract, Ptosis

model = Model(inputs=base_model.input, outputs=predictions)

# 4. Compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 5. Train
print("Starting training...")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator
)

# 6. Save the heavy model
model.save('sight_model_full.h5')
print("Training complete. Model saved as sight_model_full.h5")