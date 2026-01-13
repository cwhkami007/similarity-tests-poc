import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

filepath = "C:/Users/huichris/Downloads/all_similarity_scores_onehot.csv"
df = pd.read_csv(filepath)

print("="*70)
print("DATASET OVERVIEW")
print("="*70)
print(f"Dataset shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head(10))
print(f"\nBasic statistics:")
print(df.describe())

# Select similarity methods and transformation features
similarity_methods = ['SIFT_score', 'SSIM_score', 'CLIP_score', 'ORB_score', 
                      'VGG16_score', 'TensorFlow_score']
transformations = ['color', 'size', 'quantity', 'quality', 'mirroring', 'position']

X = df[transformations].astype(int)

print(f"\n\nTransformation Distribution:")
for trans in transformations:
    print(f"  {trans}: {X[trans].sum()} images with this transformation")

# Save results for each method
results = {}
all_feature_importances = {}

# Train separate Random Forest Models
for method in similarity_methods:
    print("\n" + "="*70)
    print(f"ANALYZING: {method}")
    print("="*70)
    
    y = df[method]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_split=3,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train, y_train)
    
    y_pred_train = rf_model.predict(X_train)
    y_pred_test = rf_model.predict(X_test)
    
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    results[method] = {
        'model': rf_model,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'y_test': y_test,
        'y_pred_test': y_pred_test
    }
    
    # Feature importance
    feature_imp = pd.DataFrame({
        'transformation': transformations,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    all_feature_importances[method] = feature_imp
    
    # Print results
    print(f"\nTraining Set Performance:")
    print(f"  R² Score: {train_r2:.4f}")
    print(f"  RMSE: {train_rmse:.4f}")
    print(f"  MAE: {train_mae:.4f}")
    
    print(f"\nTest Set Performance:")
    print(f"  R² Score: {test_r2:.4f}")
    print(f"  RMSE: {test_rmse:.4f}")
    print(f"  MAE: {test_mae:.4f}")
    
    print(f"\nFeature Importance Rankings:")
    for idx, row in feature_imp.iterrows():
        print(f"  {row['transformation']:12s}: {row['importance']:.4f}")

#Visualizations

fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

#R² Scores
ax1 = fig.add_subplot(gs[0, 0])
methods_short = [m.replace('_score', '') for m in similarity_methods]
train_r2_scores = [results[m]['train_r2'] for m in similarity_methods]
test_r2_scores = [results[m]['test_r2'] for m in similarity_methods]

x_pos = np.arange(len(methods_short))
width = 0.35
ax1.bar(x_pos - width/2, train_r2_scores, width, label='Train', alpha=0.8)
ax1.bar(x_pos + width/2, test_r2_scores, width, label='Test', alpha=0.8)
ax1.set_ylabel('R² Score')
ax1.set_title('Model Performance: R² Score by Method')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(methods_short, rotation=45, ha='right')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')
ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

#RMSE
ax2 = fig.add_subplot(gs[0, 1])
train_rmse_scores = [results[m]['train_rmse'] for m in similarity_methods]
test_rmse_scores = [results[m]['test_rmse'] for m in similarity_methods]

ax2.bar(x_pos - width/2, train_rmse_scores, width, label='Train', alpha=0.8)
ax2.bar(x_pos + width/2, test_rmse_scores, width, label='Test', alpha=0.8)
ax2.set_ylabel('RMSE')
ax2.set_title('Model Performance: RMSE by Method')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(methods_short, rotation=45, ha='right')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')


#Actual vs Predicted 
for idx, method in enumerate(similarity_methods):
    row = 1 + idx // 3
    col = idx % 3
    ax = fig.add_subplot(gs[row, col])
    
    y_test = results[method]['y_test']
    y_pred = results[method]['y_pred_test']
    
    ax.scatter(y_test, y_pred, alpha=0.6, s=50)
    
    # Perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, 
            label='Perfect Prediction')
    
    ax.set_xlabel('Actual Score')
    ax.set_ylabel('Predicted Score')
    ax.set_title(f'{method.replace("_score", "")} (R²={results[method]["test_r2"]:.3f})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.savefig('rf_multi_method_analysis.png', dpi=300, bbox_inches='tight')
print("\n\nVisualization saved as 'rf_multi_method_analysis.png'")
plt.show()

#Transformation Impact Summary 

print("\n\n" + "="*70)
print("REPORT 1: TRANSFORMATION IMPACT ACROSS ALL METHODS")
print("="*70)

# Calculate average importance for each transformation
avg_importance = {}
for trans in transformations:
    importances = [all_feature_importances[m][
        all_feature_importances[m]['transformation'] == trans
    ]['importance'].values[0] for m in similarity_methods]
    avg_importance[trans] = {
        'mean': np.mean(importances),
        'std': np.std(importances),
        'min': np.min(importances),
        'max': np.max(importances)
    }

# Sort by average importance
sorted_trans = sorted(avg_importance.items(), key=lambda x: x[1]['mean'], reverse=True)

print("\nTransformation Impact Ranking (averaged across all methods):")
print(f"{'Rank':<6} {'Transformation':<15} {'Avg Importance':<18} {'Std Dev':<12} {'Range'}")
print("-" * 70)
for rank, (trans, stats) in enumerate(sorted_trans, 1):
    print(f"{rank:<6} {trans:<15} {stats['mean']:.4f} ({stats['mean']*100:5.1f}%)  "
          f"{stats['std']:.4f}      {stats['min']:.4f} - {stats['max']:.4f}")

print("\nKey Insights:")
print(f"• Most impactful transformation: {sorted_trans[0][0]} "
      f"(avg importance: {sorted_trans[0][1]['mean']:.4f})")
print(f"• Least impactful transformation: {sorted_trans[-1][0]} "
      f"(avg importance: {sorted_trans[-1][1]['mean']:.4f})")

#Similarity Method Comparison

print("\n\n" + "="*70)
print("REPORT 2: SIMILARITY METHOD ACCURACY AND CHARACTERISTICS")
print("="*70)

# Create comparison df
comparison_df = pd.DataFrame({
    'Method': methods_short,
    'Test R²': [results[m]['test_r2'] for m in similarity_methods],
    'Test RMSE': [results[m]['test_rmse'] for m in similarity_methods],
    'Test MAE': [results[m]['test_mae'] for m in similarity_methods],
    'Train R²': [results[m]['train_r2'] for m in similarity_methods],
    'Overfit Gap': [results[m]['train_r2'] - results[m]['test_r2'] 
                    for m in similarity_methods]
}).sort_values('Test R²', ascending=False)

print("\nMethod Performance Summary (sorted by Test R²):")
print(comparison_df.to_string(index=False))

print("\n\nMethod Rankings:")
print("\nBy Predictability (Test R²):")
for idx, row in comparison_df.iterrows():
    print(f"  {comparison_df.index.get_loc(idx)+1}. {row['Method']:<12} "
          f"R² = {row['Test R²']:.4f}")

print("\nBy Precision (Test RMSE - lower is better):")
sorted_by_rmse = comparison_df.sort_values('Test RMSE')
for idx, row in sorted_by_rmse.iterrows():
    print(f"  {sorted_by_rmse.index.get_loc(idx)+1}. {row['Method']:<12} "
          f"RMSE = {row['Test RMSE']:.4f}")

print("\nBy Stability (Overfitting Gap - lower is better):")
sorted_by_overfit = comparison_df.sort_values('Overfit Gap')
for idx, row in sorted_by_overfit.iterrows():
    print(f"  {sorted_by_overfit.index.get_loc(idx)+1}. {row['Method']:<12} "
          f"Gap = {row['Overfit Gap']:.4f}")

# Most sensitive transformation
print("\n\nMost Sensitive Transformation by Method:")
for method in similarity_methods:
    top_trans = all_feature_importances[method].iloc[0]
    print(f"  {method.replace('_score', ''):<12}: {top_trans['transformation']} "
          f"(importance: {top_trans['importance']:.4f})")

print("\n\n" + "="*70)
print("RECOMMENDATIONS")
print("="*70)

best_method = comparison_df.iloc[0]['Method']
most_stable = sorted_by_overfit.iloc[0]['Method']
most_precise = sorted_by_rmse.iloc[0]['Method']

print(f"\n• Best Overall Predictability: {best_method}")
print(f"  → Highest R² score, best explains transformation impact")

print(f"\n• Most Stable Method: {most_stable}")
print(f"  → Smallest train-test gap, generalizes well")

print(f"\n• Most Precise Method: {most_precise}")
print(f"  → Lowest RMSE, most accurate predictions")

print("\n\n" + "="*70)
print("Analysis Complete!")
print("="*70)
print("\nOutputs generated:")
print("  1. Console: Detailed metrics and rankings")
print("  2. rf_multi_method_analysis.png: Comprehensive visualizations")
print("="*70)