import tensorflow as tf

saved_model_dir = 'sight_model_tf'

print(f"Loading from {saved_model_dir}...")
loaded_obj = tf.saved_model.load(saved_model_dir)

# 1. Extract the inference function
# The model object itself isn't callable, but its 'serving_default' signature is.
inference_func = loaded_obj.signatures['serving_default']

# 2. Define the Concrete Function wrapper
# We wrap the specific inference function, not the generic object.
@tf.function(input_signature=[tf.TensorSpec(shape=[1, 224, 224, 3], dtype=tf.float32)])
def runner(input_tensor):
    # We call the extracted signature. 
    # Note: Signatures return a dictionary, so we grab the specific output key.
    # The key is usually 'dense_1' or similar, but we can access it via the output values.
    outputs = inference_func(input_tensor)
    return list(outputs.values())[0] 

# 3. Get the concrete function
concrete_func = runner.get_concrete_function()

# 4. Convert
print("Converting with Fixed Shape...")
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open('sight_model_quant.tflite', 'wb') as f:
    f.write(tflite_model)

print("✅ Success! Fixed shape model generated.")