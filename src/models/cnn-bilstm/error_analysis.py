import pandas as pd
import cv2
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import numpy as np


model_name = "202412070826" # Change model name !
error_analysis_path = f"models/handwriting_recognition_torch/{model_name}/error_analysis"

# Load errors
error_df = pd.read_csv(f"{error_analysis_path}/error_analysis.csv")
print(error_df.head())

# Show errors (optional)
# for index, row in error_df.iterrows():
#     image_path = row['image_path']
#     label = row['label']
#     prediction = row['prediction']
#     cer = row['CER']
#     wer = row['WER']
#     image = cv2.imread(image_path)
#     cv2.putText(image, f"Label: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 150, 0), 2)
#     cv2.putText(image, f"Prediction: {prediction}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
#     cv2.putText(image, f"CER: {cer:.2f}, WER: {wer:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
#     cv2.imshow("Error Analysis", image)
#     cv2.waitKey(0)  # Press any key to move to next image
# cv2.destroyAllWindows()


# Character frequency
true_chars = ''.join(error_df['label'])
pred_chars = ''.join(error_df['prediction'])
true_char_count = Counter(true_chars)
pred_char_count = Counter(pred_chars)

# Total true and predicted character counts
total_true = sum(true_char_count.values())
total_pred = sum(pred_char_count.values())

# Convert to percentage
true_char_percent = {char: count / total_true * 100 for char, count in true_char_count.items()}
pred_char_percent = {char: count / total_pred * 100 for char, count in pred_char_count.items()}

# Print top 10 true and predicted characters in percentage
print("Top 10 True Characters (Percentage):")
for char, percent in sorted(true_char_percent.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"{char}: {percent:.2f}%")

print("\nTop 10 Predicted Characters (Percentage):")
for char, percent in sorted(pred_char_percent.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"{char}: {percent:.2f}%")

# Most frequent move errors
incorrect_moves = error_df[error_df['label'] != error_df['prediction']]
move_errors = incorrect_moves.groupby(['label', 'prediction']).size().reset_index(name='count')
move_errors = move_errors.sort_values(by='count', ascending=False)

# Total errors
total_errors = move_errors['count'].sum()
print(f"\nTotal Errors: {total_errors}")

# Add percentage column
move_errors['percentage'] = move_errors['count'] / total_errors * 100

# Print top 10 errors for Overleaf table
top_move_errors = move_errors.head(10)
print("\nTop 10 Move Errors (for LaTeX Table):")
print("Label & Prediction & Count & Percentage \\\\")
print("\\hline")
for _, row in top_move_errors.iterrows():
    print(f"{row['label']} & {row['prediction']} & {row['count']} & {row['percentage']:.2f}\\% \\\\")

# Plot horizontal bar chart
plt.barh(
    [f"{row['label']} -> {row['prediction']}" for _, row in top_move_errors.iterrows()],
    top_move_errors['percentage']
)
plt.xlabel("Percentage (%)")
plt.ylabel("Label -> Prediction")
plt.title("Top Misinterpreted Moves (Percentage)")
plt.gca().invert_yaxis()
plt.show()


# Load preds_and_labels_padded
preds_and_labels_padded_df = pd.read_csv(f"{error_analysis_path}/preds_and_labels_padded.csv")
print(preds_and_labels_padded_df.head())
print(preds_and_labels_padded_df.shape)

# Confusion matrix
y_true = []
y_pred = []

for _, row in preds_and_labels_padded_df.iterrows():
    for t, p in zip(row['label'], row['prediction']):
        y_true.append(t)
        y_pred.append(p)

labels = sorted(set(y_true + y_pred))

print(labels)

cm = confusion_matrix(y_true, y_pred)

cm = np.delete(cm, 3, axis=0)
cm = np.delete(cm, 3, axis=1)
labels.remove('.')

# Plotting the confusion matrix
plt.figure(figsize=(6, 6))

# Use Seaborn to create the heatmap without annotations (no values inside blocks)
sns.heatmap(cm, annot=False, fmt='d', linewidths=0.5, square=True, vmax=40, cmap='viridis',
            xticklabels=[labels[x] for x in range(0,27)], 
            yticklabels=[labels[x] for x in range(0,27)],
)

# Set labels and title
plt.xlabel("Predicted Labels [character]")
plt.ylabel("True Labels [character]")

plt.show()

