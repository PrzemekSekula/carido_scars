"""
Results Analysis Script for Model Evaluation
This script analyzes the results from evaluate_model.py and provides comprehensive insights.

Autor: Przemek Sekula
Created: 2025-01-25
Last modified: 2025-01-25
"""

# %%
# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

# %%
# Load the results data
csv_path = Path("test_results/results.csv")
if not csv_path.exists():
    raise FileNotFoundError(f"Results file not found: {csv_path}")

df = pd.read_csv(csv_path)
print(f"Loaded {len(df)} results from {csv_path}")
print(f"Columns: {list(df.columns)}")
print("\nFirst 5 rows:")
df.head()

# %%
# Basic data exploration
print("=== DATA OVERVIEW ===")
print(f"Total samples: {len(df)}")
print(f"Unique images: {df['image_name'].nunique()}")
print(f"Real class distribution:")
print(df['real_class'].value_counts().sort_index())
print(f"\nPredicted class distribution:")
print(df['predicted_class'].value_counts().sort_index())

# %%
# Convert probability to float for analysis
df['predicted_probability'] = df['predicted_probability'].astype(float)

# Basic statistics
print("=== PREDICTION CONFIDENCE STATISTICS ===")
print(f"Mean confidence: {df['predicted_probability'].mean():.4f}")
print(f"Median confidence: {df['predicted_probability'].median():.4f}")
print(f"Min confidence: {df['predicted_probability'].min():.4f}")
print(f"Max confidence: {df['predicted_probability'].max():.4f}")
print(f"Std confidence: {df['predicted_probability'].std():.4f}")

# %%
# Calculate accuracy and other metrics
correct_predictions = (df['real_class'] == df['predicted_class']).sum()
total_predictions = len(df)
accuracy = correct_predictions / total_predictions

print("=== PERFORMANCE METRICS ===")
print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Correct predictions: {correct_predictions}/{total_predictions}")

# %%
# Confusion Matrix
cm = confusion_matrix(df['real_class'], df['predicted_class'])
print("=== CONFUSION MATRIX ===")
print("True\\Pred    0    1")
print(f"0          {cm[0,0]:4d} {cm[0,1]:4d}")
print(f"1          {cm[1,0]:4d} {cm[1,1]:4d}")

# Calculate precision, recall, F1 for each class
tn, fp, fn, tp = cm.ravel()
precision_0 = tn / (tn + fn) if (tn + fn) > 0 else 0
recall_0 = tn / (tn + fp) if (tn + fp) > 0 else 0
precision_1 = tp / (tp + fp) if (tp + fp) > 0 else 0
recall_1 = tp / (tp + fn) if (tp + fn) > 0 else 0

print(f"\nClass 0 - Precision: {precision_0:.4f}, Recall: {recall_0:.4f}")
print(f"Class 1 - Precision: {precision_1:.4f}, Recall: {recall_1:.4f}")

# %%
# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['True 0', 'True 1'])
plt.title('Confusion Matrix')
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.tight_layout()
plt.show()

# %%
# Confidence distribution analysis
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(df['predicted_probability'], bins=20, alpha=0.7, edgecolor='black')
plt.xlabel('Predicted Probability')
plt.ylabel('Frequency')
plt.title('Distribution of Prediction Confidence')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
# Separate by correct/incorrect predictions
correct_mask = df['real_class'] == df['predicted_class']
plt.hist(df[correct_mask]['predicted_probability'], bins=15, alpha=0.7, 
         label='Correct', color='green', edgecolor='black')
plt.hist(df[~correct_mask]['predicted_probability'], bins=15, alpha=0.7, 
         label='Incorrect', color='red', edgecolor='black')
plt.xlabel('Predicted Probability')
plt.ylabel('Frequency')
plt.title('Confidence by Prediction Correctness')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# Analyze misclassifications
misclassified = df[df['real_class'] != df['predicted_class']]
print("=== MISCLASSIFICATION ANALYSIS ===")
print(f"Total misclassifications: {len(misclassified)}")
print(f"Misclassification rate: {len(misclassified)/len(df)*100:.2f}%")

if len(misclassified) > 0:
    print(f"\nMisclassified images:")
    print(misclassified[['image_name', 'real_class', 'predicted_class', 'predicted_probability']].to_string(index=False))
    
    print(f"\nAverage confidence of misclassifications: {misclassified['predicted_probability'].mean():.4f}")
    print(f"Average confidence of correct predictions: {df[correct_mask]['predicted_probability'].mean():.4f}")

# %%
# ROC Curve and AUC (if binary classification)
if len(df['real_class'].unique()) == 2:
    # For ROC curve, we need probabilities for the positive class
    # Assuming class 1 is positive class
    y_true = df['real_class']
    y_scores = df['predicted_probability']
    
    # If predicted class is 0, use 1 - probability as score for class 1
    y_scores_roc = np.where(df['predicted_class'] == 1, 
                           df['predicted_probability'], 
                           1 - df['predicted_probability'])
    
    auc = roc_auc_score(y_true, y_scores_roc)
    fpr, tpr, _ = roc_curve(y_true, y_scores_roc)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print(f"=== ROC ANALYSIS ===")
    print(f"AUC Score: {auc:.4f}")

# %%
# Confidence threshold analysis
print("=== CONFIDENCE THRESHOLD ANALYSIS ===")
thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
print("Threshold | Accuracy | Samples")
print("-" * 35)

for threshold in thresholds:
    high_conf_mask = df['predicted_probability'] >= threshold
    if high_conf_mask.sum() > 0:
        high_conf_accuracy = (df[high_conf_mask]['real_class'] == df[high_conf_mask]['predicted_class']).mean()
        print(f"{threshold:8.2f} | {high_conf_accuracy:8.4f} | {high_conf_mask.sum():7d}")
    else:
        print(f"{threshold:8.2f} | {'N/A':8s} | {high_conf_mask.sum():7d}")

# %%
# Summary report
print("\n" + "="*50)
print("=== FINAL SUMMARY REPORT ===")
print("="*50)
print(f"Total test samples: {len(df)}")
print(f"Overall accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Misclassifications: {len(misclassified)} ({len(misclassified)/len(df)*100:.2f}%)")
print(f"Average confidence: {df['predicted_probability'].mean():.4f}")
if len(df['real_class'].unique()) == 2:
    print(f"AUC Score: {auc:.4f}")

print(f"\nClass distribution:")
print(f"  Class 0: {df['real_class'].value_counts().get(0, 0)} samples")
print(f"  Class 1: {df['real_class'].value_counts().get(1, 0)} samples")

print(f"\nPrediction distribution:")
print(f"  Predicted 0: {df['predicted_class'].value_counts().get(0, 0)} samples")
print(f"  Predicted 1: {df['predicted_class'].value_counts().get(1, 0)} samples")

print("\nAnalysis complete! Check the generated plots for visual insights.")

  # %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Compute confusion matrix
cm = confusion_matrix(df['real_class'], df['predicted_class'])
classes = sorted(df['real_class'].unique())

# Calculate percentages
cm_percent = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100

plt.figure(figsize=(6, 5))
im = plt.imshow(cm_percent, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=100)
plt.title('Confusion Matrix (Percentages)')
plt.colorbar(im, fraction=0.046, pad=0.04)

tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes)
plt.yticks(tick_marks, classes)
plt.xlabel('Predicted label')
plt.ylabel('True label')

# Annotate with counts and percentages
thresh = cm_percent.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        count = cm[i, j]
        percent = cm_percent[i, j]
        plt.text(j, i, f'{count}\n{percent:.1f}%', 
                 ha="center", va="center", 
                 color="white" if percent > thresh else "black",
                 fontsize=10)

plt.tight_layout()
plt.show()
# %%