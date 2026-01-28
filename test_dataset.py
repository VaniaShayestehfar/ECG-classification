"""
ECG Arrhythmia Classification System with Ensemble Learning
============================================================
This system uses specialized ResNet models for PAC and PVC detection,
implements ensemble decision logic, and provides comprehensive evaluation
including ROC curves, confusion matrices, and arrhythmia visualization.

Author: ECG Analysis Team
Version: 2.0
Date: 2024
"""

import wfdb
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support, f1_score, roc_curve, auc
from wfdb import processing
from sklearn.preprocessing import label_binarize
import random

# ============================================================================
# GLOBAL CONFIGURATION AND CONSTANTS
# ============================================================================

# Model file paths (must be in working directory)
PAC_MODEL_PATH = 'ecg_expert_model_pac_resnet.h5'
PVC_MODEL_PATH = 'ecg_expert_model_pvc_resnet.h5'
SEGMENTS_FOLDER = ''  # Current directory for ECG segments

# Output directory for saving all results and plots
OUTPUT_FOLDER = 'ECG_Analysis_Results'

# Signal processing parameters
BEAT_LENGTH = 200
TARGET_FS = 250
TOLERANCE_WINDOW = 50  # Window for annotation matching
SMOOTHING_KERNEL_SIZE = 5

# Model-specific window sizes (must match training configuration)
WINDOW_PRE_PAC = 160
WINDOW_POST_PAC = 40
WINDOW_PRE_PVC = 50
WINDOW_POST_PVC = 150

# ============================================================================
# THRESHOLD CONFIGURATION
# ============================================================================
THRESHOLD_PAC = 0.8   # Probability threshold for PAC classification
THRESHOLD_PVC = 0.765 # Probability threshold for PVC classification
print(f"Classification Thresholds: PAC={THRESHOLD_PAC}, PVC={THRESHOLD_PVC}")

# ============================================================================
# CLASS MAPPING AND LABEL CONFIGURATION
# ============================================================================
FINAL_NUM_CLASSES = 3
FINAL_TARGET_NAMES = ['Normal/Other (N/Q)', 'PAC (S)', 'PVC (V)']
SYMBOL_TO_LABEL = {'N': 0, 'S': 1, 'V': 2}
LABEL_TO_SYMBOL = {0: 'N/Q', 1: 'S', 2: 'V'}

# ============================================================================
# MODEL LOADING FUNCTION
# ============================================================================

def load_models(pac_path, pvc_path):
    """
    Loads the pre-trained PAC and PVC ResNet models.
    
    Parameters:
    -----------
    pac_path : str
        Path to the PAC model (.h5 file)
    pvc_path : str
        Path to the PVC model (.h5 file)
    
    Returns:
    --------
    tuple
        (pac_model, pvc_model) or (None, None) if loading fails
    """
    try:
        # Suppress TensorFlow logging for cleaner output
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        
        # Load models without compilation for inference
        pac_model = load_model(pac_path, compile=False)
        pvc_model = load_model(pvc_path, compile=False)
        
        print("✅ Specialized models loaded successfully.")
        return pac_model, pvc_model
    except Exception as e:
        print(f"❌ Error loading models. Ensure the .h5 files exist. Error: {e}")
        return None, None

# Load models at startup
pac_model, pvc_model = load_models(PAC_MODEL_PATH, PVC_MODEL_PATH)

# ============================================================================
# SIGNAL PREPROCESSING FUNCTIONS
# ============================================================================

def preprocess_beat(segment, expected_length=BEAT_LENGTH):
    """
    Applies Z-score normalization and reshapes beat segment for model input.
    
    Parameters:
    -----------
    segment : numpy.ndarray
        Raw ECG beat segment
    expected_length : int
        Expected length of the beat segment
    
    Returns:
    --------
    numpy.ndarray or None
        Normalized and reshaped beat segment, or None if length mismatch
    """
    if len(segment) != expected_length:
        return None
    
    # Z-score normalization
    mean = np.mean(segment)
    std = np.std(segment)
    
    if std > 1e-6:
        beat_segment_normalized = (segment - mean) / std
    else:
        beat_segment_normalized = segment - mean
    
    # Reshape for model input: (batch_size, timesteps, channels)
    return beat_segment_normalized.reshape(1, expected_length, 1).astype('float32')

# ============================================================================
# ENSEMBLE PREDICTION LOGIC
# ============================================================================

def ensemble_predict(r_peak, signal, atr_indices, atr_symbols):
    """
    Executes ensemble prediction using both PAC and PVC models.
    Implements priority-based decision logic with thresholding.
    
    Parameters:
    -----------
    r_peak : int
        Index of the R-peak in the signal
    signal : numpy.ndarray
        Full ECG signal
    atr_indices : numpy.ndarray
        Annotation indices from the reference
    atr_symbols : list
        Annotation symbols from the reference
    
    Returns:
    --------
    tuple
        (predicted_label, true_label, score_array)
        - predicted_label: Final ensemble prediction (0, 1, or 2)
        - true_label: Ground truth label from annotations
        - score_array: 3-class probability scores for ROC analysis
    """
    if pac_model is None or pvc_model is None:
        # Return default scores if models aren't loaded
        return None, None, np.array([1.0, 0.0, 0.0])
    
    # Initialize true label as Normal/Other (0)
    true_label_final = 0
    
    # Find the closest annotation to determine ground truth
    diffs = np.abs(atr_indices - r_peak)
    closest_atr_index = np.argmin(diffs)
    
    if diffs[closest_atr_index] <= TOLERANCE_WINDOW:
        true_symbol = atr_symbols[closest_atr_index]
        true_label_final = SYMBOL_TO_LABEL.get(true_symbol, 0)
    
    # --------------------------------------------------------------------
    # PAC MODEL PREDICTION
    # --------------------------------------------------------------------
    start_pac = r_peak - WINDOW_PRE_PAC
    end_pac = r_peak + WINDOW_POST_PAC
    prob_pac = 0.0
    
    if start_pac >= 0 and end_pac <= len(signal):
        beat_input_pac = preprocess_beat(signal[start_pac:end_pac])
        if beat_input_pac is not None:
            try:
                prob_pac = pac_model.predict(beat_input_pac, verbose=0)[0][0]
            except Exception:
                prob_pac = 0.0
    
    # --------------------------------------------------------------------
    # PVC MODEL PREDICTION
    # --------------------------------------------------------------------
    start_pvc = r_peak - WINDOW_PRE_PVC
    end_pvc = r_peak + WINDOW_POST_PVC
    prob_pvc = 0.0
    
    if start_pvc >= 0 and end_pvc <= len(signal):
        beat_input_pvc = preprocess_beat(signal[start_pvc:end_pvc])
        if beat_input_pvc is not None:
            try:
                prob_pvc = pvc_model.predict(beat_input_pvc, verbose=0)[0][0]
            except Exception:
                prob_pvc = 0.0
    
    # --------------------------------------------------------------------
    # ENSEMBLE DECISION LOGIC
    # --------------------------------------------------------------------
    is_pac = (prob_pac >= THRESHOLD_PAC)
    is_pvc = (prob_pvc >= THRESHOLD_PVC)
    
    # Priority-based classification
    if is_pvc and is_pac:
        # Both models detect abnormality - choose the one with higher probability
        predicted_label_final = 2 if prob_pvc >= prob_pac else 1
    elif is_pvc:
        predicted_label_final = 2  # PVC detected
    elif is_pac:
        predicted_label_final = 1  # PAC detected
    else:
        predicted_label_final = 0  # Normal/Other
    
    # --------------------------------------------------------------------
    # SCORE CALCULATION FOR ROC ANALYSIS (One-vs-Rest)
    # --------------------------------------------------------------------
    # Score array: [Score_vs_N/Q, Score_vs_S, Score_vs_V]
    score_s = prob_pac  # PAC score
    score_v = prob_pvc  # PVC score
    score_nq = 1.0 - max(prob_pac, prob_pvc)  # Normal/Other score
    
    # Normalize to get valid probability distribution
    total_score = score_nq + score_s + score_v
    if total_score > 0:
        score_array = np.array([score_nq, score_s, score_v]) / total_score
    else:
        score_array = np.array([1.0, 0.0, 0.0])  # Default to Normal
    
    return predicted_label_final, true_label_final, score_array

# ============================================================================
# ROC CURVE PLOTTING FUNCTION
# ============================================================================

def plot_roc_curves(y_true, y_scores, target_names, segment_name, output_folder):
    """
    Plots micro-average and one-vs-rest ROC curves for multi-class classification.
    
    Parameters:
    -----------
    y_true : list or numpy.ndarray
        True class labels
    y_scores : numpy.ndarray
        Predicted scores for each class (shape: n_samples × n_classes)
    target_names : list
        Names of the target classes
    segment_name : str
        Name of the ECG segment for title and filename
    output_folder : str
        Directory to save the ROC plot
    """
    # Binarize true labels for ROC computation
    y_true_bin = label_binarize(y_true, classes=np.arange(len(target_names)))
    
    # Initialize dictionaries for ROC metrics
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    # Compute ROC for each class
    for i in range(len(target_names)):
        # Skip if class has no positive samples
        if np.sum(y_true_bin[:, i]) == 0:
            fpr[i], tpr[i], roc_auc[i] = np.array([0.0, 1.0]), np.array([0.0, 1.0]), 0.5
            continue
        
        try:
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        except ValueError:
            fpr[i], tpr[i], roc_auc[i] = np.array([0.0, 1.0]), np.array([0.0, 1.0]), 0.5
    
    # Compute micro-average ROC (aggregates all classes)
    try:
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_scores.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    except Exception:
        fpr["micro"], tpr["micro"], roc_auc["micro"] = np.array([0.0, 1.0]), np.array([0.0, 1.0]), 0.5
    
    # Create ROC plot
    plt.figure(figsize=(10, 8))
    
    # Plot micro-average ROC
    plt.plot(
        fpr["micro"], tpr["micro"],
        label=f'Micro-Average ROC (AUC = {roc_auc["micro"]:.2f})',
        color='deeppink', linestyle=':', linewidth=4
    )
    
    # Plot one-vs-rest ROC for each class
    colors = ['blue', 'darkorange', 'green']
    for i, color in zip(range(len(target_names)), colors):
        plt.plot(
            fpr[i], tpr[i], color=color, lw=2,
            label=f'ROC {target_names[i]} (AUC = {roc_auc[i]:.2f})'
        )
    
    # Plot chance line (diagonal)
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Chance (AUC = 0.50)')
    
    # Configure plot aesthetics
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR) / Recall')
    plt.title(f'Receiver Operating Characteristic (ROC) - Segment: {segment_name}')
    plt.legend(loc="lower right")
    
    # Save plot to file
    output_path = os.path.join(output_folder, f'{segment_name}_ROC_Curve.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ ROC curve saved: {output_path}")

# ============================================================================
# MAIN EVALUATION FUNCTION
# ============================================================================

def evaluate_full_signal_ensemble_with_cm(segment_name, output_folder, generate_plots=True):
    """
    Main evaluation function that processes an ECG segment, runs ensemble prediction,
    generates performance metrics, and creates visualizations.
    
    Parameters:
    -----------
    segment_name : str
        Name of the ECG segment (without file extension)
    output_folder : str
        Directory to save output files
    generate_plots : bool
        Whether to generate and save plots
    
    Returns:
    --------
    float or None
        Weighted F1-score of the segment, or None if evaluation fails
    """
    # Construct full file path
    segment_path_prefix = os.path.join(SEGMENTS_FOLDER, segment_name)
    
    # Logging
    if generate_plots:
        print(f"\n{'='*60}")
        print(f"Evaluating Segment: {segment_name}")
        print(f"{'='*60}")
    else:
        print(f"Re-evaluating (for plots): {segment_name}")
    
    # Check model availability
    if pac_model is None or pvc_model is None:
        return None
    
    try:
        # Load ECG record and annotations
        record = wfdb.rdrecord(segment_path_prefix, physical=True)
        annotation = wfdb.rdann(segment_path_prefix, 'atr')
    except Exception as e:
        if generate_plots:
            print(f"❌ Error loading files for {segment_name}: {e}")
        return None
    
    # Extract signal data
    signal = record.p_signal[:, 0]  # Lead I
    fs = record.fs  # Sampling frequency
    
    # --------------------------------------------------------------------
    # R-PEAK DETECTION
    # --------------------------------------------------------------------
    if generate_plots:
        try:
            # Primary detector
            qrs_indices = processing.gqrs_detect(sig=signal, fs=fs)
            print(f"Using gqrs_detect - Found {len(qrs_indices)} R-peaks.")
        except Exception:
            # Fallback detector
            qrs_indices = processing.xqrs_detect(sig=signal, fs=fs)
            print(f"Using xqrs_detect (fallback) - Found {len(qrs_indices)} R-peaks.")
    else:
        # Use faster detection for re-evaluation
        qrs_indices = processing.xqrs_detect(sig=signal, fs=fs)
    
    # --------------------------------------------------------------------
    # ENSEMBLE PREDICTION LOOP
    # --------------------------------------------------------------------
    true_labels_final = []
    predicted_labels_final = []
    scores_final = []  # For ROC analysis
    final_counts = {'N/Q': 0, 'S': 0, 'V': 0}
    
    for r_peak in qrs_indices:
        # Get ensemble prediction
        pred_label, true_label, score_array = ensemble_predict(
            r_peak, signal, annotation.sample, annotation.symbol
        )
        
        # Store results if valid
        if true_label is not None:
            true_labels_final.append(true_label)
            predicted_labels_final.append(pred_label)
            scores_final.append(score_array)
            
            # Update counts
            predicted_symbol = LABEL_TO_SYMBOL.get(pred_label)
            if predicted_symbol in final_counts:
                final_counts[predicted_symbol] += 1
    
    # Check if any beats were evaluated
    if not true_labels_final:
        if generate_plots:
            print("🛑 No evaluable beats found in the segment.")
        return None
    
    # Convert scores to numpy array for ROC analysis
    scores_final_np = np.array(scores_final)
    
    # Calculate weighted F1-score (primary performance metric)
    try:
        weighted_f1 = f1_score(true_labels_final, predicted_labels_final, 
                               average='weighted', zero_division=0)
    except ValueError:
        return None
    
    # --------------------------------------------------------------------
    # PERFORMANCE METRICS AND REPORTING (only if generating plots)
    # --------------------------------------------------------------------
    if generate_plots:
        # Compute confusion matrix
        cm = confusion_matrix(true_labels_final, predicted_labels_final, 
                              labels=np.arange(FINAL_NUM_CLASSES))
        
        # Compute per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels_final, predicted_labels_final,
            labels=np.arange(FINAL_NUM_CLASSES), average=None, zero_division=0
        )
        
        # Global accuracy
        global_accuracy = accuracy_score(true_labels_final, predicted_labels_final)
        
        # --------------------------------------------------------------------
        # PRINT COMPREHENSIVE REPORT
        # --------------------------------------------------------------------
        print("\n📊 PERFORMANCE REPORT")
        print("-" * 60)
        print(f"Weighted F1-Score: {weighted_f1:.4f}")
        print(f"Global Accuracy: {global_accuracy:.4f}")
        print(f"\nPredicted Beat Distribution:")
        print(f"  Normal/Other (N/Q): {final_counts['N/Q']}")
        print(f"  PAC (S): {final_counts['S']}")
        print(f"  PVC (V): {final_counts['V']}")
        
        print("\n📈 Confusion Matrix:")
        df_cm = pd.DataFrame(cm, index=FINAL_TARGET_NAMES, columns=FINAL_TARGET_NAMES)
        print(df_cm)
        
        print("\n🎯 Detailed Metrics per Class:")
        metrics_data = {
            'Class': FINAL_TARGET_NAMES,
            'Precision': [f'{p:.4f}' for p in precision],
            'Recall': [f'{r:.4f}' for r in recall],
            'F1-Score': [f'{f:.4f}' for f in f1],
            'Support': support
        }
        df_metrics = pd.DataFrame(metrics_data)
        print(df_metrics.to_markdown(index=False))
        
        # --------------------------------------------------------------------
        # VISUALIZATION 1: ROC CURVES
        # --------------------------------------------------------------------
        plot_roc_curves(true_labels_final, scores_final_np, 
                        FINAL_TARGET_NAMES, segment_name, output_folder)
        
        # --------------------------------------------------------------------
        # VISUALIZATION 2: CONFUSION MATRIX HEATMAP
        # --------------------------------------------------------------------
        plt.figure(figsize=(8, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=FINAL_TARGET_NAMES, 
                    yticklabels=FINAL_TARGET_NAMES)
        plt.title(f'Confusion Matrix: {segment_name}\n(Weighted F1 = {weighted_f1:.4f})')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        
        # Save confusion matrix
        cm_path = os.path.join(output_folder, f'{segment_name}_CM.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Confusion matrix saved: {cm_path}")
        
        # --------------------------------------------------------------------
        # VISUALIZATION 3: ARRHYTHMIA SAMPLE PLOTS
        # --------------------------------------------------------------------
        ZOOM_WINDOW_SECONDS = 10
        ZOOM_WINDOW_SAMPLES = int(ZOOM_WINDOW_SECONDS * fs)
        arrhythmia_plots_count = 0
        MAX_PLOTS = 2  # Limit number of plots for efficiency
        
        for r_peak in qrs_indices:
            if arrhythmia_plots_count >= MAX_PLOTS:
                break
            
            # Get probabilities for display (re-run prediction for visualization)
            prob_pac, prob_pvc = 0.0, 0.0
            
            # PAC probability
            start_pac, end_pac = r_peak - WINDOW_PRE_PAC, r_peak + WINDOW_POST_PAC
            if start_pac >= 0 and end_pac <= len(signal):
                beat_input_pac = preprocess_beat(signal[start_pac:end_pac])
                if beat_input_pac is not None:
                    try:
                        prob_pac = pac_model.predict(beat_input_pac, verbose=0)[0][0]
                    except Exception:
                        pass
            
            # PVC probability
            start_pvc, end_pvc = r_peak - WINDOW_PRE_PVC, r_peak + WINDOW_POST_PVC
            if start_pvc >= 0 and end_pvc <= len(signal):
                beat_input_pvc = preprocess_beat(signal[start_pvc:end_pvc])
                if beat_input_pvc is not None:
                    try:
                        prob_pvc = pvc_model.predict(beat_input_pvc, verbose=0)[0][0]
                    except Exception:
                        pass
            
            # Determine predicted symbol for visualization
            is_pac = (prob_pac >= THRESHOLD_PAC)
            is_pvc = (prob_pvc >= THRESHOLD_PVC)
            
            if is_pvc and is_pac:
                predicted_symbol = 'V' if prob_pvc >= prob_pac else 'S'
            elif is_pvc:
                predicted_symbol = 'V'
            elif is_pac:
                predicted_symbol = 'S'
            else:
                predicted_symbol = 'N/Q'
            
            # Only plot arrhythmia samples
            if predicted_symbol in ['S', 'V']:
                arrhythmia_plots_count += 1
                
                # Define zoom window around the R-peak
                zoom_start = max(0, r_peak - ZOOM_WINDOW_SAMPLES // 2)
                zoom_end = min(len(signal), r_peak + ZOOM_WINDOW_SAMPLES // 2)
                
                # Extract and smooth signal for visualization
                zoom_signal_raw = signal[zoom_start:zoom_end]
                kernel = np.ones(SMOOTHING_KERNEL_SIZE) / SMOOTHING_KERNEL_SIZE
                zoom_signal_smoothed = np.convolve(zoom_signal_raw, kernel, mode='same')
                
                # Create time axis
                time_axis = np.arange(zoom_start, zoom_end) / fs
                center_time = r_peak / fs
                
                # Create the plot
                plt.figure(figsize=(12, 5))
                plt.plot(time_axis, zoom_signal_smoothed, color='navy', linewidth=1.5, 
                         label='ECG Signal')
                plt.axvline(x=center_time, color='red', linestyle='--', 
                           linewidth=1.5, label='R-Peak Center')
                
                # Add annotation text
                peak_amplitude = zoom_signal_smoothed[r_peak - zoom_start]
                true_display_symbol = LABEL_TO_SYMBOL.get(true_labels_final[arrhythmia_plots_count - 1])
                
                plt.text(center_time, peak_amplitude * 1.05,
                         f'PREDICTED: {predicted_symbol}\n'
                         f'PAC Prob: {prob_pac:.3f}\n'
                         f'PVC Prob: {prob_pvc:.3f}\n'
                         f'TRUE: {true_display_symbol}',
                         color='darkred', fontsize=9, ha='center',
                         bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
                
                # Configure plot
                plt.title(f'Detected Arrhythmia: {predicted_symbol}\n'
                         f'Segment: {segment_name} | Time: {center_time:.2f}s')
                plt.xlabel('Time (seconds)')
                plt.ylabel('Amplitude (mV)')
                plt.grid(True, linestyle=':', alpha=0.5)
                plt.legend(loc='upper right')
                plt.tight_layout()
                
                # Save arrhythmia plot
                arrh_path = os.path.join(output_folder, f'{segment_name}_Arrh_{arrhythmia_plots_count}.png')
                plt.savefig(arrh_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"  ✅ Arrhythmia plot {arrhythmia_plots_count} saved: {arrh_path}")
    
    return weighted_f1

# ============================================================================
# BATCH EVALUATION AND EXECUTION
# ============================================================================

# List of ECG segments to evaluate
ALL_SEGMENTS_TO_EVALUATE = ["106"]

print(f"\n🎯 Batch Evaluation Setup")
print(f"Number of segments: {len(ALL_SEGMENTS_TO_EVALUATE)}")
print(f"Output folder: {OUTPUT_FOLDER}")

# Create output directory if it doesn't exist
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
    print(f"📁 Created output directory: {OUTPUT_FOLDER}")
else:
    print(f"📁 Using existing output directory: {OUTPUT_FOLDER}")

# Initialize results storage
results_list = []

# Check model availability before proceeding
if pac_model is None or pvc_model is None:
    print("\n❌ Model loading failed. Cannot proceed with evaluation.")
    print("Please ensure the following files exist in the working directory:")
    print(f"  - {PAC_MODEL_PATH}")
    print(f"  - {PVC_MODEL_PATH}")
else:
    # --------------------------------------------------------------------
    # BATCH EVALUATION LOOP
    # --------------------------------------------------------------------
    for segment_name in ALL_SEGMENTS_TO_EVALUATE:
        print(f"\n{'='*60}")
        print(f"Processing Segment: {segment_name}")
        print(f"{'='*60}")
        
        # Run evaluation with visualization
        weighted_f1 = evaluate_full_signal_ensemble_with_cm(
            segment_name, OUTPUT_FOLDER, generate_plots=True
        )
        
        # Store result if successful
        if weighted_f1 is not None:
            results_list.append({
                'Segment': segment_name,
                'Weighted_F1_Score': weighted_f1
            })
    
    # --------------------------------------------------------------------
    # FINAL SUMMARY AND RESULTS COMPILATION
    # --------------------------------------------------------------------
    if results_list:
        # Create results DataFrame
        df_results = pd.DataFrame(results_list)
        
        # Find best performing segment
        best_segment = df_results.loc[df_results['Weighted_F1_Score'].idxmax()]
        best_segment_name = best_segment['Segment']
        best_f1_score = best_segment['Weighted_F1_Score']
        
        # Save results to CSV
        csv_path = os.path.join(OUTPUT_FOLDER, 'F1_Score_Summary.csv')
        df_results.to_csv(csv_path, index=False)
        print(f"\n✅ Results summary saved: {csv_path}")
        
        # --------------------------------------------------------------------
        # FINAL SUMMARY DISPLAY
        # --------------------------------------------------------------------
        print("\n" + "="*80)
        print("🏆 EVALUATION COMPLETE - PERFORMANCE SUMMARY")
        print("="*80)
        print("\n📈 Segment Performance Rankings:")
        print(df_results.sort_values('Weighted_F1_Score', ascending=False).to_markdown(index=False))
        
        print(f"\n🎖️  BEST PERFORMING SEGMENT:")
        print(f"   Segment: {best_segment_name}")
        print(f"   Weighted F1-Score: {best_f1_score:.4f}")
        print(f"\n📊 This segment achieved the highest balanced performance")
        print(f"   across all classes (Normal, PAC, PVC).")
        
        # --------------------------------------------------------------------
        # REGENERATE PLOTS FOR BEST SEGMENT (FINAL VERSIONS)
        # --------------------------------------------------------------------
        print(f"\n🔄 Generating final plots for best segment: {best_segment_name}")
        evaluate_full_signal_ensemble_with_cm(best_segment_name, OUTPUT_FOLDER, generate_plots=True)
        
        print("\n" + "="*80)
        print("📁 ALL OUTPUT FILES SAVED IN:")
        print(f"   {os.path.abspath(OUTPUT_FOLDER)}")
        print("="*80)
        print("\nGenerated files include:")
        print("  • ROC curves (micro-average and per-class)")
        print("  • Confusion matrix heatmaps")
        print("  • Arrhythmia sample visualizations")
        print("  • Performance summary CSV")
        print("="*80)
        
    else:
        print("\n⚠️  No valid results were collected.")
        print("   Please check your ECG data files and ensure they are in the correct format.")

print("\n✨ ECG Analysis Pipeline Complete.")