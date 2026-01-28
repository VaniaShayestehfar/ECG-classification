import wfdb
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, Activation, Add, Input
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from wfdb import processing
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import class_weight
import time
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU Memory Growth set to True.")
    except RuntimeError as e:
        print(e)

TRAIN_LIST_FILE = 'train_set_list.csv'
VAL_LIST_FILE = 'validation_set_list.csv'
TEST_LIST_FILE = 'test_set_list.csv'
SEGMENTS_FOLDER = 'Top_10_Segments_For_Analysis'

WINDOW_PRE = 50 
WINDOW_POST = 150 
BEAT_LENGTH = WINDOW_PRE + WINDOW_POST
TARGET_FS = 250
BATCH_SIZE = 64
SUBSET_RATIO = 0.04

TARGET_CLASS = 'PVC'
POSITIVE_SYMBOL = ['V']
POSITIVE_LABEL = 1
NEGATIVE_LABEL = 0
BINARY_TARGET_NAMES = ['Non-PVC (0)', 'PVC (1)']

LOAD_SYMBOL_TO_BINARY_LABEL = {s: NEGATIVE_LABEL for s in ['N', 'S', 'Q']}
for s in POSITIVE_SYMBOL:
    LOAD_SYMBOL_TO_BINARY_LABEL[s] = POSITIVE_LABEL


def load_beats_from_segments(df_segments_subset):
    all_beats = []
    all_labels = []

    for index, row in df_segments_subset.iterrows():
        segment_name = row['Segment_Name']
        record_path_prefix = os.path.join(SEGMENTS_FOLDER, segment_name)

        try:
            record = wfdb.rdrecord(record_path_prefix, physical=True)
            annotation = wfdb.rdann(record_path_prefix, 'atr')
            signal = record.p_signal[:, 0]
            fs_raw = record.fs

            if fs_raw != TARGET_FS:
                continue

            for sample_index, symbol in zip(annotation.sample, annotation.symbol):
                label = LOAD_SYMBOL_TO_BINARY_LABEL.get(symbol, NEGATIVE_LABEL)

                start = sample_index - WINDOW_PRE
                end = sample_index + WINDOW_POST
                
                if start >= 0 and end <= len(signal):
                    beat_segment = signal[start:end]
                    
                    if len(beat_segment) == BEAT_LENGTH:
                        mean = np.mean(beat_segment)
                        std = np.std(beat_segment)
                        if std > 1e-6:
                            beat_segment = (beat_segment - mean) / std
                        else:
                             beat_segment = beat_segment - mean

                        all_beats.append(beat_segment)
                        all_labels.append(label)

        except Exception as e:
            pass
            
    return np.array(all_beats), np.array(all_labels)


try:
    df_train_full = pd.read_csv(TRAIN_LIST_FILE)
    df_val_full = pd.read_csv(VAL_LIST_FILE)
    df_test_full = pd.read_csv(TEST_LIST_FILE)
    print(f"Loaded {len(df_train_full)} training segments, {len(df_val_full)} validation, {len(df_test_full)} test segments.")
except FileNotFoundError:
    print("❌ Error: CSV list files not found. Exiting.")
    exit()

df_train_subset = df_train_full.sample(frac=SUBSET_RATIO, random_state=42)
df_val_subset = df_val_full.sample(frac=SUBSET_RATIO, random_state=42)
df_test_subset = df_test_full.sample(frac=SUBSET_RATIO, random_state=42)

print(f"\nUsing {SUBSET_RATIO*100:.2f}% subset: {len(df_train_subset)} train, {len(df_val_subset)} val, {len(df_test_subset)} test segments.")

print(f"Starting to load Train subset into RAM for {TARGET_CLASS} expert...")
X_train_raw, y_train_raw = load_beats_from_segments(df_train_subset)
print(f"Starting to load Validation subset into RAM...")
X_val_raw, y_val_raw = load_beats_from_segments(df_val_subset)
print(f"Starting to load Test subset into RAM...")
X_test_raw, y_test_raw = load_beats_from_segments(df_test_subset)


if X_train_raw.size == 0 or len(np.unique(y_train_raw)) < 2:
    print("🛑 Critical Error: Training subset is empty or only contains one class. Exiting.")
    exit()

X_train = X_train_raw.reshape(X_train_raw.shape[0], X_train_raw.shape[1], 1).astype('float32')

weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(y_train_raw),
    y=y_train_raw
)
class_weights_dict = dict(enumerate(weights))

PVC_BOOST_FACTOR = 1.5
if POSITIVE_LABEL in class_weights_dict:
    class_weights_dict[POSITIVE_LABEL] *= PVC_BOOST_FACTOR
    print(f"‼️ PVC (Class 1) weight was boosted by {PVC_BOOST_FACTOR}x: {class_weights_dict[POSITIVE_LABEL]:.4f}")
else:
    print("⚠️ Warning: Positive class not found in training data to apply weight boost.")

validation_data_tuple = None
if X_val_raw.size > 0:
    X_val = X_val_raw.reshape(X_val_raw.shape[0], X_val_raw.shape[1], 1).astype('float32')
    validation_data_tuple = (X_val, y_val_raw)

print("-" * 70)
print(f"✅ Data Preparation Complete for {TARGET_CLASS} Expert.")
print(f"Final Training Beats Count: {len(X_train)}")
print(f"Non-PVC Count (0): {np.sum(y_train_raw == NEGATIVE_LABEL)}")
print(f"PVC Count (1): {np.sum(y_train_raw == POSITIVE_LABEL)}")
print(f"⚖️ Calculated Class Weights (0: Non-PVC, 1: PVC): {class_weights_dict}")
print("-" * 70)


def residual_block(input_tensor, filters, kernel_size, stage, block, strides=1):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    input_filters = input_tensor.shape.as_list()[-1]
    
    projection_shortcut = (input_filters != filters)

    x = Conv1D(filters, kernel_size, padding='same', name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=-1, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv1D(filters, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=-1, name=bn_name_base + '2b')(x)
    
    if projection_shortcut:
        shortcut = Conv1D(filters, 1, strides=1, name=conv_name_base + '1')(input_tensor)
        shortcut = BatchNormalization(axis=-1, name=bn_name_base + '1')(shortcut)
    else:
        shortcut = input_tensor

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

def create_resnet_cnn_model(input_shape):
    
    input_layer = Input(shape=input_shape)
    x = Conv1D(filters=64, kernel_size=15, strides=2, padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.2)(x)

    x = residual_block(x, filters=64, kernel_size=5, stage=1, block='a')
    x = residual_block(x, filters=64, kernel_size=5, stage=1, block='b')
    x = MaxPooling1D(pool_size=2)(x)
    
    x = residual_block(x, filters=128, kernel_size=5, stage=2, block='a')
    x = residual_block(x, filters=128, kernel_size=5, stage=2, block='b')
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.3)(x)
    
    x = residual_block(x, filters=256, kernel_size=3, stage=3, block='a')
    x = residual_block(x, filters=256, kernel_size=3, stage=3, block='b')

    x = Flatten()(x)
    x = Dense(100, activation='relu')(x)
    x = Dropout(0.4)(x)
    
    output_layer = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=input_layer, outputs=output_layer)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

input_shape = (BEAT_LENGTH, 1)
model = create_resnet_cnn_model(input_shape)
MODEL_PATH = f'ecg_expert_model_{TARGET_CLASS.lower()}_resnet.h5'

callbacks = [
    EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1)
]

print(f"\nStarting training for {TARGET_CLASS} Expert (ResNet-like)...")

start_time = time.time()
history = model.fit(
    X_train, y_train_raw, 
    epochs=50,
    batch_size=BATCH_SIZE,
    validation_data=validation_data_tuple, 
    callbacks=callbacks,
    class_weight=class_weights_dict, 
    verbose=1
)
end_time = time.time()

model.save(MODEL_PATH)
print(f"\nTraining finished in {(end_time - start_time):.2f} seconds.")
print(f"Model saved to {MODEL_PATH}")


def evaluate_subset_test_data(X_test_raw, y_test_raw, model_path, threshold=0.65):
    
    try:
        model = load_model(model_path, compile=False)
    except Exception as e:
        print(f"❌ Evaluation failed: Could not load model from {model_path}. Error: {e}")
        return
    
    if X_test_raw.size == 0:
        print("❌ Evaluation failed: Test subset is empty.")
        return

    X_test = X_test_raw.reshape(X_test_raw.shape[0], X_test_raw.shape[1], 1).astype('float32')

    y_pred_probs = model.predict(X_test, batch_size=512, verbose=0)
    
    y_pred_labels = (y_pred_probs > threshold).astype(int).flatten()

    y_true_labels = y_test_raw.astype(int)

    global_accuracy = accuracy_score(y_true_labels, y_pred_labels)
    cm = confusion_matrix(y_true_labels, y_pred_labels, labels=[NEGATIVE_LABEL, POSITIVE_LABEL]) 
    report = classification_report(y_true_labels, y_pred_labels, target_names=BINARY_TARGET_NAMES, zero_division=0)
    
    f1_pvc = f1_score(y_true_labels, y_pred_labels, pos_label=POSITIVE_LABEL, zero_division=0)
    
    print("\n" + "=" * 50)
    print(f"--- Final Performance Metrics (ResNet-like Model) ---")
    print(f"*** Using Decision Threshold: {threshold} ***")
    print(f"Total Test Beats: {len(X_test)}")
    print(f"PVC (1) Beats in Test Set: {np.sum(y_true_labels == POSITIVE_LABEL)}")
    print(f"✅ PVC Expert Model F1-Score (Target Metric for Class '1'): {f1_pvc:.4f}")
    print(f"Global Classification Accuracy: {global_accuracy:.4f}")
    print("\n--- Detailed Classification Report ---")
    print(report)
    print("=" * 50)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=BINARY_TARGET_NAMES, yticklabels=BINARY_TARGET_NAMES)
    plt.title(f'Confusion Matrix for {TARGET_CLASS} Expert (Test Set, Threshold={threshold})')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

evaluate_subset_test_data(X_test_raw, y_test_raw, MODEL_PATH, threshold=0.65)