# Import libraries
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import joblib
import sys
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

print("✅ Libraries imported successfully.")

# --- 1. User Input Section ---

print("\n--- 1. Configure Your Model ---")

# 1a. Choose Feature File
print("Which feature file do you want to use?")
print("  [1] data/BTC-USD-1h-features.csv (All Data)")
print("  [2] data/BTC-USD-1h-BEAR-features.csv (Bear Market)")
print("  [3] data/BTC-USD-1h-BULL-features.csv (Bull Market)")
file_choice = input("Enter choice (1, 2, or 3): ")

if file_choice == '1':
    FEATURES_FILE = 'data/BTC-USD-1h-features.csv'
    output_model_file = 'models/btc_xgboost_model.joblib'
elif file_choice == '2':
    FEATURES_FILE = 'data/BTC-USD-1h-BEAR-features.csv'
    output_model_file = 'models/bear_xgboost_model.joblib'
elif file_choice == '3':
    FEATURES_FILE = 'data/BTC-USD-1h-BULL-features.csv'
    output_model_file = 'models/bull_xgboost_model.joblib'
else:
    print("Invalid choice. Exiting.")
    sys.exit()

# 1b. Use SMOTE?
smote_choice = input("Use SMOTE to balance data? (y/n): ").lower()
use_smote = True if smote_choice == 'y' else False

# 1c. Set n_iter
try:
    n_iter_input = int(input("Enter number of search iterations (e.g., 25): "))
except ValueError:
    print("Invalid number. Defaulting to 25.")
    n_iter_input = 25

# 1d. Set Confidence Threshold
try:
    confidence_input = float(input("Enter confidence threshold for report (e.g., 0.65): "))
except ValueError:
    print("Invalid number. Defaulting to 0.65.")
    confidence_input = 0.65

print("--- Configuration Complete ---")
print(f"  File:    {FEATURES_FILE}")
print(f"  SMOTE:   {use_smote}")
print(f"  n_iter:  {n_iter_input}")
print(f"  Threshold: {confidence_input}")
print("------------------------------\n")


# --- 2. Load Data ---

if not os.path.exists(FEATURES_FILE):
    raise FileNotFoundError(f"{FEATURES_FILE} does not exist! Make sure feature engineering ran correctly.")

df = pd.read_csv(FEATURES_FILE, parse_dates=True, index_col='datetime')

print(f"✅ Feature data loaded. Shape: {df.shape}")
print("Columns:", df.columns.tolist())


# --- 3. Feature Engineering & Splitting ---

print("⚙️ Adding safe lag features...")
# Add lag features (1, 2, and 3 periods back)
for lag in [1, 2, 3]:
    df[f"target_lag{lag}"] = df["target"].shift(lag)

# Handle NaNs caused by shifting
df.fillna(0, inplace=True)
print("✅ Added target lag features.")

# Define X and y AFTER feature engineering
y = df['target']
features_to_drop = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'target']
X = df.drop(columns=features_to_drop, errors='ignore')

print(f"✅ Feature matrix X shape: {X.shape}")
print(f"✅ Target vector y shape: {y.shape}")

# Split the Data (Chronological)
split_point = int(len(df) * 0.8)
X_train, X_test = X[:split_point], X[split_point:]
y_train, y_test = y[:split_point], y[split_point:]

print(f"✅ Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
if len(X_train) == 0 or len(X_test) == 0:
    raise ValueError("Training or testing set is empty! Check your feature engineering and CSV data.")


# --- 4. Pre-processing (SMOTE & Encoding) ---

# Encode labels to 0,1,2 for XGBoost
y_train_xgb = y_train + 1
y_test_xgb = y_test + 1
print("✅ Labels encoded for XGBoost (0,1,2).")

if use_smote:
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train_xgb)
    print("✅ SMOTE applied → Balanced training set:", np.bincount(y_train_bal))
else:
    X_train_bal, y_train_bal = X_train, y_train_xgb
    print("ℹ️ SMOTE skipped.")


# --- 5. Hyperparameter Tuning ---

xgb = XGBClassifier(
    random_state=42,
    n_jobs=-1,
    objective='multi:softprob',
    eval_metric='mlogloss'
)

print("✅ XGBoost classifier initialized.")

param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
}

random_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_grid,
    n_iter=n_iter_input,  # Using user input
    scoring='f1_macro',
    cv=3,
    verbose=4,
    random_state=42,
    n_jobs=-1
)

print("🚀 Starting hyperparameter tuning...")
with joblib.parallel_backend('loky', inner_max_num_threads=1):
    random_search.fit(X_train_bal, y_train_bal)

print("✅ Hyperparameter tuning complete!")
print("Best Parameters:", random_search.best_params_)


# --- 6. Evaluation ---

model = random_search.best_estimator_

# Predictions
predictions_raw = model.predict(X_test)
predictions = predictions_raw - 1  # convert back to (-1,0,1)

print("\n📊 Classification Report (All Predictions):")
print(classification_report(y_test, predictions, digits=3))

# Probabilities for confidence filtering
probabilities = model.predict_proba(X_test)

results_df = pd.DataFrame(index=y_test.index)
results_df['true_label'] = y_test
results_df['prediction'] = predictions
results_df['confidence_sell'] = probabilities[:,0]
results_df['confidence_hold'] = probabilities[:,1]
results_df['confidence_buy'] = probabilities[:,2]
results_df['confidence_of_prediction'] = probabilities.max(axis=1)

CONFIDENCE_THRESHOLD = confidence_input  # Using user input
results_df['high_conf_prediction'] = np.where(
    results_df['confidence_of_prediction'] > CONFIDENCE_THRESHOLD,
    results_df['prediction'],
    0
)

print(f"\n--- Classification Report (Confidence > {CONFIDENCE_THRESHOLD*100}%) ---")
print(classification_report(
    results_df['true_label'],
    results_df['high_conf_prediction'],
    labels=[-1,0,1],
    target_names=['Sell (-1)','Hold (0)','Buy (1)']
))


# --- 6b. Plot Feature Importance ---

print("\n--- Feature Importance ---")
try:
    # Get feature names from your training data
    feature_names = X_train_bal.columns
    
    # Get the importance scores from the best model
    importances = model.feature_importances_
    
    # Create a DataFrame for easy plotting
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    
    # Plot the top 15 features
    plt.figure(figsize=(10, 8))
    feat_imp.head(15).plot(kind='barh', title='Top 15 Feature Importances')
    plt.gca().invert_yaxis() # Display most important at the top
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.show()
    print("Feature importance plot generated.")

except Exception as e:
    print(f"Could not generate feature importance plot: {e}")


# --- 6c. Prepare and Save Data for Backtesting ---

print("\n--- Preparing data for backtest... ---")

# The results_df already has the correct test set index.
# We just need to join the original price data (open, high, low, close)
# from the main 'df' so the backtester knows the prices.
backtest_cols = ['open', 'high', 'low', 'close']

# Use .join() to merge the price data onto our results_df using the shared datetime index
backtest_df = results_df.join(df[backtest_cols])

# Re-order columns to be more logical
final_cols = ['open', 'high', 'low', 'close', 'true_label', 'prediction', 'high_conf_prediction', 
              'confidence_sell', 'confidence_hold', 'confidence_buy']

# Filter for only the columns that exist to avoid KeyErrors
final_cols_exist = [col for col in final_cols if col in backtest_df.columns]
backtest_df = backtest_df[final_cols_exist]

# Save to a new, dedicated file
BACKTEST_FILE_PATH = 'data/backtest_results.csv'
try:
    os.makedirs('data', exist_ok=True) # Ensure 'data' folder exists
    backtest_df.to_csv(BACKTEST_FILE_PATH)
    print(f"✅ Backtest-ready data saved successfully to: {BACKTEST_FILE_PATH}")
    print(backtest_df.head())
except Exception as e:
    print(f"\n--- ERROR ---")
    print(f"Could not save backtest data file: {e}")


# --- 7. Save Model ---

output_model_folder = 'models'
os.makedirs(output_model_folder, exist_ok=True)

# Using dynamic output file name
joblib.dump(model, output_model_file)
print(f"\n✅ XGBoost Model saved successfully to: {output_model_file}")
