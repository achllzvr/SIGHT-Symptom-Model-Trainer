import tensorflow as tf
import numpy as np

# Load your trained model
model = tf.keras.models.load_model('sight_model_full.h5')

# Create Converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# --- QUANTIZATION MAGIC ---
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# This ensures the math uses integers (fast) instead of floats (slow)
def representative_data_gen():
    # We generate random noise just to tell the converter the data shape
    # In a real scenario, use real images here for better accuracy
    for _ in range(100):
        yield [np.random.rand(1, 224, 224, 3).astype(np.float32)]

converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8  # IMPORTANT: Inputs must be 0-255 integers
converter.inference_output_type = tf.uint8

tflite_model = converter.convert()

# Save
with open('sight_model_quant.tflite', 'wb') as f:
    f.write(tflite_model)

print("Conversion complete. File saved: sight_model_quant.tflite")