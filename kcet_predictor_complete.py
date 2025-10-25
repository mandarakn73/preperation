# ============================================================================
# File: app.py
# Main application file for KCET College Predictor
# ============================================================================

import pandas as pd
import os
from dotenv import load_dotenv
import gradio as gr
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import joblib
import numpy as np

# --- Configuration ---
load_dotenv()

# Load CET cutoff data
try:
    df = pd.read_csv("CET-CUTOFF2025.csv")
    print(f"Loaded {len(df)} college records from CET-CUTOFF2025.csv")
except FileNotFoundError:
    print("Error: CET-CUTOFF2025.csv not found.")
    df = pd.DataFrame()

# Geolocation setup
BANGALORE_COORD = (12.9716, 77.5946)
geolocator = Nominatim(user_agent="kcet_predictor")

# Load Extra Trees model
extra_trees_model = None
try:
    extra_trees_model = joblib.load("extra_trees_model.joblib")
    print("Extra Trees model loaded successfully.")
except FileNotFoundError:
    print("Warning: 'extra_trees_model.joblib' not found. Run train_model.py first.")
except Exception as e:
    print(f"Error loading ML model: {e}")

# --- Helper Functions ---

def get_distance_from_bangalore(location):
    """Calculate distance from Bangalore for a given location."""
    try:
        loc = geolocator.geocode(location, timeout=5)
        if loc:
            distance = geodesic(BANGALORE_COORD, (loc.latitude, loc.longitude)).km
            return round(distance, 1)
    except Exception as e:
        print(f"Geocoding error for {location}: {e}")
    return 100.0  # Default distance if geocoding fails


# --- Agent Classes ---

class StudentInputAgent:
    """Processes and validates student input data."""
    
    def get_student_info(self, rank, caste):
        """
        Validates and returns student information.
        
        Args:
            rank: Student's KCET rank
            caste: Student's caste category
            
        Returns:
            Dictionary with validated student info
        """
        return {
            "rank": int(rank), 
            "caste": caste.upper()
        }


class MLCollegePredictor:
    """
    Predicts the best colleges using the Extra Trees model.
    The model predicts admission probability/score for each college.
    """
    
    def __init__(self, model, college_data):
        self.model = model
        self.college_data = college_data
        self.location_cache = {}  # Cache distances to avoid repeated geocoding
    
    def _get_cached_distance(self, location):
        """Get distance with caching to improve performance."""
        if location not in self.location_cache:
            self.location_cache[location] = get_distance_from_bangalore(location)
        return self.location_cache[location]
    
    def _prepare_features(self, student_rank, college_row, caste_category):
        """
        Prepare feature vector for a single college prediction.
        
        Features:
        - student_rank: Student's KCET rank
        - college_distance_km: Distance from Bangalore
        - caste_cutoff: Historical cutoff for the category
        - rank_difference: How much better/worse student is compared to cutoff
        """
        location = college_row.get("Location", "Bangalore")
        distance = self._get_cached_distance(location)
        caste_cutoff = college_row.get(caste_category, 0)
        
        # Create feature dictionary
        features = {
            "student_rank": student_rank,
            "college_distance_km": distance,
            "caste_cutoff": caste_cutoff,
            "rank_difference": caste_cutoff - student_rank
        }
        
        return pd.DataFrame([features])
    
    def predict(self, student_rank, student_caste):
        """
        Predict and rank colleges for a given student.
        
        Args:
            student_rank: Student's KCET rank
            student_caste: Student's caste category
            
        Returns:
            DataFrame with top 5 predicted colleges
        """
        if self.model is None:
            return self._fallback_prediction(student_rank, student_caste)
        
        if self.college_data.empty:
            print("No college data available.")
            return pd.DataFrame()
        
        predictions = []
        
        # Generate predictions for all colleges
        for idx, row in self.college_data.iterrows():
            try:
                # Check if student is eligible (rank <= cutoff)
                caste_cutoff = row.get(student_caste, 0)
                if caste_cutoff == 0 or student_rank > caste_cutoff:
                    continue  # Skip ineligible colleges
                
                # Prepare features for prediction
                features = self._prepare_features(student_rank, row, student_caste)
                
                # Get prediction score (higher = better match)
                predicted_score = self.model.predict(features)[0]
                
                predictions.append({
                    "CETCode": row.get("CETCode", "N/A"),
                    "College": row.get("College", "Unknown"),
                    "Location": row.get("Location", "Unknown"),
                    "Branch": row.get("Branch", "Unknown"),
                    "Cutoff": int(caste_cutoff),
                    "PredictedScore": round(predicted_score, 2)
                })
                
            except Exception as e:
                print(f"Prediction error for college {row.get('College', 'Unknown')}: {e}")
                continue
        
        if not predictions:
            print("No eligible colleges found for the given criteria.")
            return pd.DataFrame()
        
        # Sort by predicted score (descending) and return top 5
        predictions_df = pd.DataFrame(predictions)
        predictions_df = predictions_df.sort_values("PredictedScore", ascending=False)
        
        return predictions_df.head(5)
    
    def _fallback_prediction(self, student_rank, student_caste):
        """
        Fallback method when ML model is not available.
        Uses simple cutoff-based filtering.
        """
        print("Using fallback prediction (cutoff-based filtering)")
        
        if self.college_data.empty:
            return pd.DataFrame()
        
        # Filter colleges where student rank <= cutoff
        eligible = self.college_data[
            self.college_data.get(student_caste, pd.Series(dtype=float)) >= student_rank
        ].copy()
        
        if eligible.empty:
            return pd.DataFrame()
        
        # Add distance-based scoring
        eligible["Distance"] = eligible["Location"].apply(self._get_cached_distance)
        eligible["Score"] = (eligible[student_caste] - student_rank) / eligible["Distance"]
        
        # Sort by score and return top 5
        eligible = eligible.sort_values("Score", ascending=False)
        
        result = eligible.head(5)[[
            "CETCode", "College", "Location", "Branch", student_caste
        ]].copy()
        result.columns = ["CETCode", "College", "Location", "Branch", "Cutoff"]
        
        return result


# --- Gradio Interface ---

def predict_colleges(rank, caste):
    """
    Main prediction function for Gradio interface.
    
    Args:
        rank: Student's KCET rank
        caste: Student's caste category
        
    Returns:
        DataFrame with predicted colleges
    """
    try:
        # Get and validate student info
        student_agent = StudentInputAgent()
        student_info = student_agent.get_student_info(rank, caste)
        
        # Initialize predictor
        ml_predictor = MLCollegePredictor(extra_trees_model, df)
        
        # Get predictions
        results = ml_predictor.predict(
            student_info["rank"], 
            student_info["caste"]
        )
        
        if results.empty:
            return pd.DataFrame({
                "Message": ["No colleges found matching your criteria. Try a different rank or category."]
            })
        
        return results
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        return pd.DataFrame({
            "Error": [f"An error occurred: {str(e)}"]
        })


# --- Gradio UI ---

with gr.Blocks(css="""
    .gr-block-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 10px;
    }
    .gr-button {
        background-color: #0055ff !important;
        color: white !important;
        font-weight: bold !important;
        border-radius: 5px !important;
        padding: 10px 20px !important;
    }
    .gr-button:hover {
        background-color: #0044cc !important;
    }
    .gr-title {
        text-align: center;
        color: white;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .gr-description {
        text-align: center;
        color: white;
        margin-bottom: 20px;
    }
""") as demo:

    gr.Markdown('<h1 class="gr-title">🎓 KCET College Predictor</h1>')
    gr.Markdown('<p class="gr-description">Powered by Extra Trees Machine Learning Model</p>')
    
    with gr.Row():
        with gr.Column():
            rank_input = gr.Slider(
                label="📊 KCET Rank",
                minimum=1,
                maximum=100000,
                step=1,
                value=5000,
                info="Enter your KCET rank"
            )
            
            caste_input = gr.Dropdown(
                label="👤 Category",
                choices=[
                    "1G", "1K", "1R",
                    "2AG", "2AK", "2AR",
                    "2BG", "2BK", "2BR",
                    "3AG", "3AK", "3AR",
                    "3BG", "3BK", "3BR",
                    "GM", "GMK", "GMR",
                    "SCG", "SCK", "SCR",
                    "STG", "STK", "STR"
                ],
                value="1G",
                info="Select your category"
            )
            
            predict_btn = gr.Button("🔮 Predict Colleges", size="lg")
    
    gr.Markdown("### 📋 Top College Predictions")
    
    output_box = gr.Dataframe(
        headers=["CETCode", "College", "Location", "Branch", "Cutoff", "PredictedScore"],
        datatype=["str", "str", "str", "str", "number", "number"],
        label="Your Best Matches",
        wrap=True
    )
    
    gr.Markdown("""
    **Note:** 
    - Predictions are based on historical data and ML model analysis
    - Higher PredictedScore indicates better match
    - Cutoff shows the last year's closing rank for your category
    """)
    
    predict_btn.click(
        fn=predict_colleges,
        inputs=[rank_input, caste_input],
        outputs=output_box
    )


if __name__ == "__main__":
    demo.launch(share=False)


# ============================================================================
# File: train_model.py
# Script to train the Extra Trees model
# ============================================================================

"""
TRAINING SCRIPT - Run this first to generate the model

Usage:
    python train_model.py

This will create 'extra_trees_model.joblib' file needed by app.py
"""

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
