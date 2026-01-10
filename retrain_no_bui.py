import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import pickle

# Force retrain with 8 features (no BUI)
features = ['Temperature', ' RH', ' Ws', 'Rain ', 'FFMC', 'DMC', 'DC', 'ISI']

df = pd.read_csv('data/Algerian_forest_fires_dataset.csv')
# convert to numeric
for col in features:
    df[col] = pd.to_numeric(df[col], errors='coerce')
# drop rows with missing
df = df.dropna(subset=features + ['Classes  '])
X = df[features]
y = (df['Classes  '].str.strip() == 'fire').astype(int)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = Ridge(alpha=1.0, random_state=42)
model.fit(X_scaled, y)

with open('models/scaler.pkl','wb') as f:
    pickle.dump(scaler,f)
with open('models/ridge.pkl','wb') as f:
    pickle.dump(model,f)

print('Retrain complete. scaler n_features_in_ =', getattr(scaler,'n_features_in_', None), 'mean_len=', len(getattr(scaler,'mean_', [])))
print('feature_names_in_=', getattr(scaler,'feature_names_in_', None))