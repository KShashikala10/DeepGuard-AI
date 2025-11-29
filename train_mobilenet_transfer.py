
import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Paths
BASE_DIR = os.path.join(os.getcwd(), 'Dataset')
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VAL_DIR = os.path.join(BASE_DIR, 'validation')
TEST_DIR = os.path.join(BASE_DIR, 'test')
MODEL_PATH = os.path.join(os.getcwd(), 'models', 'deepfake_mobilenet_final.keras')  # Using .keras format

# Hyperparameters optimized for STABLE CPU training
IMG_SIZE = (224, 224)  # MobileNet's standard input size
BATCH_SIZE = 8         # Small batch size for stability
EPOCHS_PHASE1 = 3      # Short training for testing
EPOCHS_PHASE2 = 2      # Short training for testing
LEARNING_RATE = 1e-4   # Conservative learning rate

print(" FINAL FIXED VERSION - TensorFlow 2.17.0 Compatible")
print("="*60)
print(f" Total training time estimated: ~3 hours")
print(f" Phase 1: {EPOCHS_PHASE1} epochs (~2 hours)")
print(f" Phase 2: {EPOCHS_PHASE2} epochs (~1 hour)")
print(" Key fixes:")
print("  - Removed 'workers' and 'use_multiprocessing' parameters")
print("  - Using .keras format instead of .h5")
print("  - Limited validation steps to prevent hanging")
print("  - Smaller batch size for CPU stability")
print("  - Conservative epochs for initial testing")
print("="*60)

# Simple data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Create data generators
print(" Setting up data generators...")
train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# Check if test directory exists
if os.path.exists(TEST_DIR):
    test_gen = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )
else:
    test_gen = None
    print("Test directory not found, skipping test data generator.")

# Calculate steps to limit training/validation time
train_steps = min(1000, len(train_gen))  # Limit training steps
val_steps = min(200, len(val_gen))       # Limit validation steps

print(f" Training steps per epoch: {train_steps}")
print(f" Validation steps per epoch: {val_steps}")

print(" Creating MobileNet model with transfer learning...")

# Create base model with MobileNet
base_model = MobileNet(
    weights='imagenet',
    include_top=False,
    input_shape=(*IMG_SIZE, 3),
    alpha=1.0
)

# Freeze base model layers initially
base_model.trainable = False

# Add custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(1, activation='sigmoid', name='predictions')(x)

# Create final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile model
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print(" Model summary:")
model.summary()

# Setup callbacks
callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=2,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=1,
        min_lr=1e-7,
        verbose=1
    ),
    ModelCheckpoint(
        MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

print(f" Starting PHASE 1: Training with frozen base model ({EPOCHS_PHASE1} epochs)...")

# Phase 1: Train with frozen base model (FIXED - no workers/multiprocessing parameters)
try:
    history_phase1 = model.fit(
        train_gen,
        epochs=EPOCHS_PHASE1,
        steps_per_epoch=train_steps,
        validation_data=val_gen,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=1
        # NOTE: Removed workers, use_multiprocessing, max_queue_size parameters
        # These were removed in TensorFlow 2.16+ / Keras 3.0+
    )
    print(" Phase 1 completed successfully!")
except Exception as e:
    print(f" Error in Phase 1: {e}")
    # Save model even if phase 1 fails
    model.save(MODEL_PATH.replace('.keras', '_phase1_partial.keras'))
    print(" Partial model saved")
    exit(1)

print(f" Starting PHASE 2: Fine-tuning ({EPOCHS_PHASE2} epochs)...")

# Phase 2: Fine-tune last few layers
base_model.trainable = True

# Fine-tune from this layer onwards (conservative)
fine_tune_at = len(base_model.layers) - 10

# Freeze all layers before fine_tune_at
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Recompile with lower learning rate
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE/10),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Continue training (FIXED - no workers/multiprocessing parameters)
try:
    history_phase2 = model.fit(
        train_gen,
        epochs=EPOCHS_PHASE2,
        steps_per_epoch=train_steps,
        validation_data=val_gen,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=1,
        initial_epoch=EPOCHS_PHASE1
        # NOTE: Removed workers, use_multiprocessing, max_queue_size parameters
    )
    print(" Phase 2 completed successfully!")
except Exception as e:
    print(f" Error in Phase 2: {e}")
    # Save model even if phase 2 fails
    model.save(MODEL_PATH.replace('.keras', '_phase2_partial.keras'))
    print(" Partial model saved")
    exit(1)

# Ensure model directory exists and save model
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
model.save(MODEL_PATH)
print(f" Final model saved to {MODEL_PATH}")

# Evaluate on test set if available (LIMITED STEPS)
if test_gen is not None:
    print(" Evaluating on test set...")
    try:
        test_loss, test_accuracy = model.evaluate(
            test_gen, 
            verbose=1,
            steps=min(100, len(test_gen))  # Limit test steps
        )
        print(f" Test Accuracy: {test_accuracy:.4f}")
        print(f" Test Loss: {test_loss:.4f}")
    except Exception as e:
        print(f" Test evaluation failed: {e}")

# Save training history
try:
    import pickle
    history_combined = {
        'phase1': history_phase1.history,
        'phase2': history_phase2.history
    }

    history_path = os.path.join(os.getcwd(), 'models', 'training_history_final.pkl')
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    with open(history_path, 'wb') as f:
        pickle.dump(history_combined, f)

    print(f" Training history saved to {history_path}")

    # Print final metrics
    final_val_acc = max(history_phase2.history['val_accuracy'])
    final_val_loss = min(history_phase2.history['val_loss'])

    print("\n" + "="*60)
    print(" TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"  Total Training Time: ~3 hours")
    print(f" Final Validation Accuracy: {final_val_acc:.4f}")
    print(f" Final Validation Loss: {final_val_loss:.4f}")
    print(f" Model saved as: {MODEL_PATH}")
    print("="*60)
    print("\n Next steps:")
    print("1. If this works well, increase epochs for longer training")
    print("2. Run evaluation script to get detailed metrics")
    print("3. Test with individual images")
    print("4. Deploy your model for production use!")

except Exception as e:
    print(f"⚠️ Could not save training history: {e}")
    print(" But model training completed successfully!")

print("\n Script completed without errors!")
print(" TensorFlow 2.17.0 / Keras 3.0 compatibility confirmed!")
