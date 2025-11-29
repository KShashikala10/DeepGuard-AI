import tensorflow as tf
import os

# Load and analyze the model
model_path = 'models/deepfake_keras3_compatible.keras'
model = tf.keras.models.load_model(model_path)

print("=" * 60)
print("DEEPGUARD AI MODEL ANALYSIS")
print("=" * 60)

print(f"Model Input Shape: {model.input_shape}")
print(f"Model Output Shape: {model.output_shape}")
print(f"Total Parameters: {model.count_params():,}")

print("\nModel Architecture Summary:")
print("-" * 40)
for i, layer in enumerate(model.layers):
    if hasattr(layer, 'units'):
        print(f"{i+1:2d}. {layer.name:<25} {layer.__class__.__name__:<20} Units: {layer.units}")
    elif hasattr(layer, 'filters'):
        print(f"{i+1:2d}. {layer.name:<25} {layer.__class__.__name__:<20} Filters: {layer.filters}")
    else:
        print(f"{i+1:2d}. {layer.name:<25} {layer.__class__.__name__}")

print("\nDetailed Model Summary:")
print("-" * 40)
model.summary()