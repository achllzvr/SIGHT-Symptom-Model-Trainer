import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# 1. SETUP & DATA
# We explicitly define the input shape here
IMG_SHAPE = (224, 224, 3)
BATCH_SIZE = 32
train_dir = 'sight_dataset'
val_dir = 'sight_dataset'

print("Preparing Data Generators...")
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    horizontal_flip=True,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# 2. BUILD MODEL
# We use an explicit Input layer to lock the shape
inputs = Input(shape=IMG_SHAPE)
base_model = MobileNetV2(input_tensor=inputs, weights='imagenet', include_top=False)
base_model.trainable = False 

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
# Automatically detect number of classes (should be 5)
predictions = Dense(len(train_generator.class_indices), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 3. TRAIN
print("Starting Training...")
model.fit(
    train_generator,
    epochs=10, # 10 is enough for 95%+ accuracy usually
    validation_data=val_generator
)

# 4. SAVE (CRITICAL STEP)
# We save as a SavedModel Folder, which handles signatures better than .h5
print("Saving Model...")
tf.saved_model.save(model, 'sight_model_tf')
print("✅ Model saved to folder: 'sight_model_tf'")