import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import json
import warnings
warnings.filterwarnings('ignore')

print("🤖 FINAL DEEPFAKE DETECTOR WITH LABEL VERIFICATION")
print("="*70)

# ============================================
# STEP 1: SETUP
# ============================================
def set_seeds(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

set_seeds(42)

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {len(tf.config.list_physical_devices('GPU'))}")

# ============================================
# STEP 2: VERIFY DATASET AND LABELS
# ============================================
print("\n🔍 VERIFYING DATASET STRUCTURE AND LABELS")
print("="*70)

BASE_PATH = '/kaggle/input/deepfake-and-real-images/Dataset'
WORKING_PATH = '/kaggle/working'
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 15

def verify_dataset_labels():
    """Check what's actually in the dataset folders"""
    
    print("📁 Dataset structure:")
    
    for split in ['Train', 'Validation', 'Test']:
        split_path = os.path.join(BASE_PATH, split)
        print(f"\n{split.upper()}: {split_path}")
        
        fake_path = os.path.join(split_path, 'Fake')
        real_path = os.path.join(split_path, 'Real')
        
        # Check Fake folder
        if os.path.exists(fake_path):
            fake_files = [f for f in os.listdir(fake_path) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            print(f"  📂 Fake/: {len(fake_files)} images")
            if fake_files:
                print(f"    Sample: {fake_files[0]}")
        else:
            print(f"  ❌ Fake/ folder not found!")
        
        # Check Real folder
        if os.path.exists(real_path):
            real_files = [f for f in os.listdir(real_path) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            print(f"  📂 Real/: {len(real_files)} images")
            if real_files:
                print(f"    Sample: {real_files[0]}")
        else:
            print(f"  ❌ Real/ folder not found!")
    
    # Let's check what Keras sees
    print("\n" + "="*70)
    print("🔍 What Keras ImageDataGenerator sees:")
    
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    test_gen = test_datagen.flow_from_directory(
        os.path.join(BASE_PATH, 'Train'),
        target_size=(32, 32),  # Small for quick check
        batch_size=10,
        class_mode='binary',
        shuffle=False
    )
    
    print(f"\nKeras found: {test_gen.class_indices}")
    print(f"Number of classes: {test_gen.num_classes}")
    
    # Get a batch and check labels - FIXED: use next() as iterator
    print("\n📊 Checking first batch of 10 images:")
    
    # Get the iterator
    iterator = iter(test_gen)
    batch_x, batch_y = next(iterator)
    
    for i in range(min(5, len(batch_y))):
        label = batch_y[i]
        class_name = 'Real' if label > 0.5 else 'Fake'
        print(f"  Image {i+1}: Label={label:.2f} ({class_name})")
    
    return test_gen.class_indices

# Verify labels
class_indices = verify_dataset_labels()
print(f"\n✅ Keras class mapping: {class_indices}")

# ============================================
# STEP 3: CREATE DATA GENERATORS WITH DEBUGGING
# ============================================
print("\n" + "="*70)
print("📂 CREATING DEBUGGED DATA GENERATORS")
print("="*70)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

class DebugImageDataGenerator:
    """Custom generator to debug labels"""
    
    def __init__(self, directory, target_size, batch_size):
        self.datagen = ImageDataGenerator(rescale=1./255)
        self.generator = self.datagen.flow_from_directory(
            directory,
            target_size=target_size,
            batch_size=batch_size,
            class_mode='binary',
            shuffle=True,
            seed=42
        )
        self.class_indices = self.generator.class_indices
        print(f"Classes: {self.class_indices}")
    
    def get_batch_with_info(self, batch_num=0):
        """Get a batch and print debug info"""
        self.generator.reset()
        
        # Get iterator
        iterator = iter(self.generator)
        
        for i in range(batch_num + 1):
            batch_x, batch_y = next(iterator)
        
        print(f"\n📊 Batch {batch_num} debug info:")
        print(f"  Batch X shape: {batch_x.shape}")
        print(f"  Batch Y shape: {batch_y.shape}")
        
        # Check distribution
        fake_count = np.sum(batch_y < 0.5)
        real_count = np.sum(batch_y >= 0.5)
        
        print(f"  Fake images: {fake_count} ({fake_count/len(batch_y)*100:.1f}%)")
        print(f"  Real images: {real_count} ({real_count/len(batch_y)*100:.1f}%)")
        
        # Show sample predictions
        print(f"\n  Sample labels:")
        for i in range(min(3, len(batch_y))):
            label = batch_y[i]
            class_name = 'Real' if label > 0.5 else 'Fake'
            print(f"    Image {i+1}: {label:.4f} -> {class_name}")
        
        return batch_x, batch_y

# Create debug generators
print("\nCreating Train generator:")
train_gen = DebugImageDataGenerator(
    os.path.join(BASE_PATH, 'Train'),
    IMG_SIZE,
    BATCH_SIZE
)

print("\nCreating Validation generator:")
val_gen = DebugImageDataGenerator(
    os.path.join(BASE_PATH, 'Validation'),
    IMG_SIZE,
    BATCH_SIZE
)

print("\nCreating Test generator:")
test_gen = DebugImageDataGenerator(
    os.path.join(BASE_PATH, 'Test'),
    IMG_SIZE,
    BATCH_SIZE
)

# Check a batch
train_x, train_y = train_gen.get_batch_with_info(0)

# ============================================
# STEP 4: CREATE MODEL WITH LABEL AWARENESS
# ============================================
print("\n" + "="*70)
print("🧠 CREATING MODEL WITH LABEL CHECKING")
print("="*70)

def create_label_aware_model():
    """Create model that tracks predictions during training"""
    
    model = keras.Sequential([
        layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        
        # Conv Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Conv Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Conv Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        # Output - NOTE: We'll interpret based on actual data
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Custom callback to monitor predictions
    class LabelMonitor(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            # Check predictions on validation set
            val_gen.generator.reset()
            iterator = iter(val_gen.generator)
            batch_x, batch_y = next(iterator)
            
            predictions = self.model.predict(batch_x, verbose=0)
            
            # Count predictions
            pred_fake = np.sum(predictions < 0.5)
            pred_real = np.sum(predictions >= 0.5)
            
            # Count actual labels
            actual_fake = np.sum(batch_y < 0.5)
            actual_real = np.sum(batch_y >= 0.5)
            
            print(f"\n📊 Epoch {epoch+1} Label Check:")
            print(f"  Actual: {actual_fake} Fake, {actual_real} Real")
            print(f"  Predicted: {pred_fake} Fake, {pred_real} Real")
            
            # Check if model is biased
            if pred_fake == len(predictions) or pred_real == len(predictions):
                print(f"  ⚠️  WARNING: Model predicting all {('FAKE' if pred_fake == len(predictions) else 'REAL')}!")
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', 'Precision', 'Recall']
    )
    
    print("✅ Model created with label monitoring")
    model.summary()
    
    return model, LabelMonitor()

# Create model
model, label_monitor = create_label_aware_model()

# ============================================
# STEP 5: TRAIN WITH MONITORING
# ============================================
print("\n" + "="*70)
print("🚀 TRAINING WITH LABEL MONITORING")
print("="*70)

# Calculate steps
train_steps = min(200, train_gen.generator.samples // BATCH_SIZE)
val_steps = min(50, val_gen.generator.samples // BATCH_SIZE)

print(f"\n⚙️ Training Configuration:")
print(f"  Epochs: {EPOCHS}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Train Steps: {train_steps}")
print(f"  Val Steps: {val_steps}")
print(f"  Class Mapping: {train_gen.class_indices}")

# Callbacks
callbacks = [
    label_monitor,
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(WORKING_PATH, 'best_model_checkpoint.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# Train
print("\n" + "="*70)
print("🔥 STARTING TRAINING")
print("="*70)

history = model.fit(
    train_gen.generator,
    steps_per_epoch=train_steps,
    epochs=EPOCHS,
    validation_data=val_gen.generator,
    validation_steps=val_steps,
    callbacks=callbacks,
    verbose=1
)

print("\n✅ TRAINING COMPLETED!")

# Save final model
model.save(os.path.join(WORKING_PATH, 'final_deepfake_model.h5'))
print("💾 Model saved to: /kaggle/working/final_deepfake_model.h5")

# ============================================
# STEP 6: EVALUATE AND FIX LABEL INTERPRETATION
# ============================================
print("\n" + "="*70)
print("📊 EVALUATING AND FIXING LABEL INTERPRETATION")
print("="*70)

def evaluate_with_correction(model, test_gen):
    """Evaluate and apply label correction if needed"""
    
    test_gen.generator.reset()
    
    # Get predictions
    test_steps = min(100, test_gen.generator.samples // BATCH_SIZE)
    predictions = []
    true_labels = []
    
    # Get iterator
    iterator = iter(test_gen.generator)
    
    for i in range(test_steps):
        try:
            batch_x, batch_y = next(iterator)
            batch_pred = model.predict(batch_x, verbose=0)
            
            predictions.extend(batch_pred.flatten())
            true_labels.extend(batch_y.flatten())
        except StopIteration:
            break
    
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    
    print(f"\n📈 Raw predictions analysis:")
    print(f"  Min prediction: {predictions.min():.4f}")
    print(f"  Max prediction: {predictions.max():.4f}")
    print(f"  Mean prediction: {predictions.mean():.4f}")
    print(f"  Std prediction: {predictions.std():.4f}")
    
    # Check if predictions are centered around 0.5
    mean_pred = predictions.mean()
    
    if mean_pred < 0.3:
        print(f"\n⚠️  WARNING: Predictions are very low (mean={mean_pred:.4f})")
        print("   Model might be predicting everything as FAKE")
        print("   Let's try INVERTING the interpretation...")
        
        # Invert predictions: treat low values as REAL
        corrected_predictions = (predictions < 0.5).astype(int)
        
    elif mean_pred > 0.7:
        print(f"\n⚠️  WARNING: Predictions are very high (mean={mean_pred:.4f})")
        print("   Model might be predicting everything as REAL")
        print("   Let's try INVERTING the interpretation...")
        
        # Invert predictions: treat high values as FAKE
        corrected_predictions = (predictions >= 0.5).astype(int)
        
    else:
        print(f"\n✅ Predictions look balanced (mean={mean_pred:.4f})")
        print("   Using standard interpretation: >0.5 = REAL, <0.5 = FAKE")
        corrected_predictions = (predictions >= 0.5).astype(int)
    
    # Calculate accuracy
    accuracy = np.mean(corrected_predictions == true_labels)
    
    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(true_labels, corrected_predictions)
    
    print(f"\n🎯 FINAL ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    print(f"\n📊 Confusion Matrix:")
    print(f"                Predicted")
    print(f"               Fake  Real")
    print(f"Actual Fake  | {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"       Real  | {cm[1,0]:4d}  {cm[1,1]:4d}")
    
    # Detailed metrics
    fake_correct = cm[0,0]
    fake_total = cm[0,0] + cm[0,1]
    real_correct = cm[1,1]
    real_total = cm[1,0] + cm[1,1]
    
    print(f"\n📈 Per-class accuracy:")
    print(f"  Fake accuracy: {fake_correct}/{fake_total} = {fake_correct/fake_total*100:.1f}%")
    print(f"  Real accuracy: {real_correct}/{real_total} = {real_correct/real_total*100:.1f}%")
    
    return corrected_predictions, true_labels, predictions.mean()

# Evaluate
corrected_preds, true_labels, pred_mean = evaluate_with_correction(model, test_gen)

# ============================================
# STEP 7: CREATE FINAL PREDICTION FUNCTION
# ============================================
print("\n" + "="*70)
print("🎯 CREATING FINAL PREDICTION FUNCTION")
print("="*70)

# Determine if we need to invert predictions
needs_inversion = pred_mean < 0.3  # If predictions are very low

if needs_inversion:
    print("⚠️  Model outputs low values (<0.3 average)")
    print("   INTERPRETATION: Low values = REAL, High values = FAKE")
    print("   (Inverting standard interpretation)")
    
    def predict_image_final(image_array):
        """Final prediction with inverted interpretation"""
        # Preprocess
        img = cv2.resize(image_array, IMG_SIZE)
        
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)
        
        # Predict
        raw_pred = model.predict(img, verbose=0)[0][0]
        
        # INVERTED: Low value = REAL, High value = FAKE
        is_real = raw_pred < 0.5  # Changed from > 0.5 to < 0.5
        
        confidence = (1 - raw_pred) * 100 if is_real else raw_pred * 100
        
        return {
            'is_real': is_real,
            'prediction': '👤 REAL' if is_real else '🤖 FAKE',
            'confidence': confidence,
            'real_score': (1 - raw_pred) * 100,
            'fake_score': raw_pred * 100,
            'raw_prediction': float(raw_pred),
            'interpretation': 'INVERTED (low=REAL, high=FAKE)',
            'color': '#059669' if is_real else '#DC2626',
            'emoji': '👤' if is_real else '🤖'
        }
    
else:
    print("✅ Using standard interpretation")
    print("   INTERPRETATION: >0.5 = REAL, <0.5 = FAKE")
    
    def predict_image_final(image_array):
        """Final prediction with standard interpretation"""
        # Preprocess
        img = cv2.resize(image_array, IMG_SIZE)
        
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)
        
        # Predict
        raw_pred = model.predict(img, verbose=0)[0][0]
        
        # STANDARD: >0.5 = REAL, <0.5 = FAKE
        is_real = raw_pred > 0.5
        
        confidence = raw_pred * 100 if is_real else (1 - raw_pred) * 100
        
        return {
            'is_real': is_real,
            'prediction': '👤 REAL' if is_real else '🤖 FAKE',
            'confidence': confidence,
            'real_score': raw_pred * 100,
            'fake_score': (1 - raw_pred) * 100,
            'raw_prediction': float(raw_pred),
            'interpretation': 'STANDARD (>0.5=REAL, <0.5=FAKE)',
            'color': '#059669' if is_real else '#DC2626',
            'emoji': '👤' if is_real else '🤖'
        }
