import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv('models/croprotation.csv')

# Preprocess the data
label_encoders = {}
categorical_cols = ['Crop Name', 'Soil Type', 'Season', 'Preferred Next Crop']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Define features and target
X = df.drop('Preferred Next Crop', axis=1)
y = df['Preferred Next Crop']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save model and label encoders
joblib.dump(model, 'crop_rotation_model.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')

print("Model trained and saved successfully!")