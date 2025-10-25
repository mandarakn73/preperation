# ============================================================================
# File: train_model.py
# Script to train the Extra Trees model
# Save this as train_model.py
# ============================================================================

import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
from geopy.geocoders import Nominatim
from geopy.distance import geodesic

print("=" * 60)
print("KCET College Predictor - Extra Trees Model Training")
print("=" * 60)

# Configuration
BANGALORE_COORD = (12.9716, 77.5946)
geolocator = Nominatim(user_agent="kcet_model_trainer")

# Load the cutoff data
try:
    df = pd.read_csv("CET-CUTOFF2025.csv")
    print(f"\n✓ Loaded {len(df)} college records from CET-CUTOFF2025.csv")
except FileNotFoundError:
    print("\n✗ Error: CET-CUTOFF2025.csv not found!")
    print("Please ensure the file exists in the current directory.")
    exit(1)

print("\nColumns in dataset:", df.columns.tolist())

# Define caste categories
CASTE_CATEGORIES = [
    "1G", "1K", "1R",
    "2AG", "2AK", "2AR",
    "2BG", "2BK", "2BR",
    "3AG", "3AK", "3AR",
    "3BG", "3BK", "3BR",
    "GM", "GMK", "GMR",
    "SCG", "SCK", "SCR",
    "STG", "STK", "STR"
]

# Filter caste categories that exist in the dataset
available_categories = [cat for cat in CASTE_CATEGORIES if cat in df.columns]
print(f"\n✓ Found {len(available_categories)} caste categories in dataset")
print(f"Categories: {', '.join(available_categories)}")


def get_distance_from_bangalore(location):
    """Calculate distance from Bangalore."""
    try:
        loc = geolocator.geocode(location, timeout=5)
        if loc:
            distance = geodesic(BANGALORE_COORD, (loc.latitude, loc.longitude)).km
            return round(distance, 1)
    except Exception as e:
        pass
    return 100.0  # Default distance


print("\n" + "=" * 60)
print("Step 1: Calculating distances from Bangalore...")
print("=" * 60)

# Add distance column (with caching to speed up)
location_distances = {}
distances = []

for idx, location in enumerate(df['Location']):
    if location not in location_distances:
        dist = get_distance_from_bangalore(location)
        location_distances[location] = dist
        print(f"  [{idx+1}/{len(df)}] {location}: {dist} km")
    distances.append(location_distances[location])

df['distance_from_bangalore'] = distances

print("\n" + "=" * 60)
print("Step 2: Generating training data...")
print("=" * 60)

# Generate synthetic training data
training_data = []

for idx, row in df.iterrows():
    college_name = row['College']
    distance = row['distance_from_bangalore']
    
    for caste in available_categories:
        cutoff = row.get(caste, 0)
        
        if cutoff > 0:  # Only use if there's a valid cutoff
            # Generate multiple samples around the cutoff rank
            
            # Excellent rank (much better than cutoff) - High probability
            excellent_rank = max(1, int(cutoff * 0.5))
            training_data.append({
                'student_rank': excellent_rank,
                'college_distance_km': distance,
                'caste_cutoff': cutoff,
                'rank_difference': cutoff - excellent_rank,
                'target_score': 95 + np.random.uniform(-5, 5)
            })
            
            # Good rank (better than cutoff)
            good_rank = max(1, int(cutoff * 0.75))
            training_data.append({
                'student_rank': good_rank,
                'college_distance_km': distance,
                'caste_cutoff': cutoff,
                'rank_difference': cutoff - good_rank,
                'target_score': 80 + np.random.uniform(-5, 5)
            })
            
            # Marginal rank (just at cutoff)
            marginal_rank = int(cutoff * 0.95)
            training_data.append({
                'student_rank': marginal_rank,
                'college_distance_km': distance,
                'caste_cutoff': cutoff,
                'rank_difference': cutoff - marginal_rank,
                'target_score': 60 + np.random.uniform(-5, 5)
            })
            
            # Borderline rank (slightly worse than cutoff)
            borderline_rank = int(cutoff * 1.05)
            training_data.append({
                'student_rank': borderline_rank,
                'college_distance_km': distance,
                'caste_cutoff': cutoff,
                'rank_difference': cutoff - borderline_rank,
                'target_score': 40 + np.random.uniform(-5, 5)
            })
            
            # Poor rank (much worse than cutoff)
            poor_rank = int(cutoff * 1.5)
            training_data.append({
                'student_rank': poor_rank,
                'college_distance_km': distance,
                'caste_cutoff': cutoff,
                'rank_difference': cutoff - poor_rank,
                'target_score': 10 + np.random.uniform(-5, 5)
            })

training_df = pd.DataFrame(training_data)
print(f"\n✓ Generated {len(training_df)} training samples")
print(f"\nTraining data shape: {training_df.shape}")
print("\nSample of training data:")
print(training_df.head(10))

print("\n" + "=" * 60)
print("Step 3: Training Extra Trees Model...")
print("=" * 60)

# Prepare features and target
feature_columns = ['student_rank', 'college_distance_km', 'caste_cutoff', 'rank_difference']
X = training_df[feature_columns]
y = training_df['target_score']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Train Extra Trees model
model = ExtraTreesRegressor(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

print("\nTraining model...")
model.fit(X_train, y_train)

# Evaluate model
print("\n" + "=" * 60)
print("Step 4: Evaluating Model Performance...")
print("=" * 60)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

train_mae = mean_absolute_error(y_train, y_pred_train)
test_mae = mean_absolute_error(y_test, y_pred_test)
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

print(f"\n📊 Training Performance:")
print(f"   MAE: {train_mae:.2f}")
print(f"   R² Score: {train_r2:.4f}")

print(f"\n📊 Test Performance:")
print(f"   MAE: {test_mae:.2f}")
print(f"   R² Score: {test_r2:.4f}")

# Feature importance
print("\n📈 Feature Importance:")
for feature, importance in zip(feature_columns, model.feature_importances_):
    print(f"   {feature}: {importance:.4f}")

# Save the model
print("\n" + "=" * 60)
print("Step 5: Saving Model...")
print("=" * 60)

try:
    joblib.dump(model, "extra_trees_model.joblib")
    print("\n✓ Model saved successfully as 'extra_trees_model.joblib'")
except Exception as e:
    print(f"\n✗ Error saving model: {e}")

print("\n" + "=" * 60)
print("Training Complete!")
print("=" * 60)
print("\nYou can now use this model with your KCET prediction app.")
print("The model file 'extra_trees_model.joblib' is ready to use.")
print("\n" + "=" * 60)

# Test prediction example
print("\n🧪 Testing a sample prediction...")
sample_student_rank = 5000
sample_distance = 50
sample_cutoff = 10000

sample_features = pd.DataFrame([{
    'student_rank': sample_student_rank,
    'college_distance_km': sample_distance,
    'caste_cutoff': sample_cutoff,
    'rank_difference': sample_cutoff - sample_student_rank
}])

sample_score = model.predict(sample_features)[0]
print(f"\nFor a student with rank {sample_student_rank}")
print(f"College cutoff: {sample_cutoff}")
print(f"Distance: {sample_distance} km")
print(f"Predicted Score: {sample_score:.2f}")
print("\n" + "=" * 60)