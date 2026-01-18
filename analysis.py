
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("husl")

# Load results
csv_path = Path("test_results/results.csv")
if not csv_path.exists():
    raise FileNotFoundError(f"Results file not found: {csv_path}")

df = pd.read_csv(csv_path)
print(f"Loaded {len(df)} results.")

# Ensure consistency
df['predicted_probability'] = df['predicted_probability'].astype(float)

# Metrics
real = df['real_class']
pred = df['predicted_class']
probs = df['predicted_probability']

# Adjust probs for ROC (if pred class is 0, prob for 1 is 1-prob)
# Assuming binary classification where class 1 is positive
# Current probability is "confidence of predicted class"
# We need "probability of class 1"
probs_class_1 = np.where(pred == 1, probs, 1 - probs)


acc = accuracy_score(real, pred)
prec = precision_score(real, pred) # weighted for imbalanced? or binary?
rec = recall_score(real, pred)
f1 = f1_score(real, pred)

print("Latex Table Metrics:")
print(f"Accuracy & {acc:.4f} \\\\")
print(f"Precision & {prec:.4f} \\\\")
print(f"Recall & {rec:.4f} \\\\")
print(f"F1 Score & {f1:.4f} \\\\")

print ('WEIGHTED METRICS:')
prec = precision_score(real, pred, average='weighted') # weighted for imbalanced? or binary?
rec = recall_score(real, pred, average='weighted')
f1 = f1_score(real, pred, average='weighted')

print(f"Weighted Precision & {prec:.4f} \\\\")
print(f"Weighted Recall & {rec:.4f} \\\\")
print(f"Weighted F1 Score & {f1:.4f} \\\\")

# Confusion Matrix
cm = confusion_matrix(real, pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 16},
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix')
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.tight_layout()
plt.savefig('paper/confusion_matrix.png', dpi=300)
print("Saved confusion_matrix.png")
plt.close()

# ROC Curve
if len(real.unique()) == 2:
    auc = roc_auc_score(real, probs_class_1)
    fpr, tpr, _ = roc_curve(real, probs_class_1)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig('paper/roc_curve.png', dpi=300)
    print(f"Saved roc_curve.png (AUC={auc:.4f})")
    print(f"AUC & {auc:.4f} \\\\")
    plt.close()

# Class Distribution Info for Text
print("Class counts:", df['real_class'].value_counts().to_dict())
