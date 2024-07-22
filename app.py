from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
import ast
import logging
from fastapi.middleware.cors import CORSMiddleware


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the pre-trained KNN model and scaler
with open('knn_model.pkl', 'rb') as model_file:
    knn = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Load nutritional data
nutritional_data = pd.read_excel('nutritional_data_1.xlsx')

# Clean the nutritional data
nutritional_data.columns = [
    'Food and Serving', 'Calories', 'Total Fat', 'Sodium', 'Potassium',
    'Total Carbo-hydrate', 'Dietary Fiber', 'Sugars', 'Protein',
    'Vitamin A', 'Vitamin C', 'Calcium', 'Iron'
]

# Ensure all relevant columns are numeric, and handle non-numeric values
numeric_columns = [
    'Calories', 'Total Fat', 'Sodium', 'Potassium', 'Total Carbo-hydrate',
    'Dietary Fiber', 'Sugars', 'Protein', 'Vitamin A', 'Vitamin C',
    'Calcium', 'Iron'
]

for column in numeric_columns:
    nutritional_data[column] = pd.to_numeric(nutritional_data[column], errors='coerce')

# Drop rows with NaN values in numeric columns
nutritional_data = nutritional_data.dropna(subset=numeric_columns)

# Define the FastAPI app
app = FastAPI()


# Configure CORS
origins = [
    "http://localhost",
    "http://localhost:8080",
    "https://health-care-app-zeta.vercel.app/"# Replace with your front-end domain
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the request body model
class UserProfile(BaseModel):
    Disliked_Foods: str
    Health_Objectives: str

def get_preferred_foods(user_profile, main_df):
    disliked_foods = ast.literal_eval(user_profile.Disliked_Foods)
    logger.info("Disliked foods: %s", disliked_foods)
    logger.info("Food and Serving column values: %s", main_df['Food and Serving'].tolist())

    preferred_foods = main_df[~main_df['Food and Serving'].isin(disliked_foods)]
    logger.info("Preferred foods after filtering disliked: %s", preferred_foods)

    if preferred_foods.empty:
        # Fallback: If preferred foods is empty, return the original DataFrame
        logger.warning("Preferred foods is empty after filtering. Returning the original DataFrame as fallback.")
        return main_df
    return preferred_foods

def adjust_query_point(query_point, health_objective):
    if health_objective == 'Digestive Issues':
        query_point[0, numeric_columns.index('Dietary Fiber')] *= 2
    elif health_objective == 'High blood pressure':
        query_point[0, numeric_columns.index('Sodium')] *= 0.5
    elif health_objective == 'Heart Health':
        query_point[0, numeric_columns.index('Total Fat')] *= 0.5
        query_point[0, numeric_columns.index('Potassium')] *= 2
    elif health_objective == 'Weight Management':
        query_point[0, numeric_columns.index('Calories')] *= 0.5
        query_point[0, numeric_columns.index('Total Fat')] *= 0.5
    elif health_objective == 'Skin Health':
        query_point[0, numeric_columns.index('Vitamin A')] *= 2
        query_point[0, numeric_columns.index('Vitamin C')] *= 2
    elif health_objective == 'Immune system support':
        query_point[0, numeric_columns.index('Vitamin C')] *= 2
        query_point[0, numeric_columns.index('Vitamin A')] *= 2
    elif health_objective == 'Bone Health':
        query_point[0, numeric_columns.index('Calcium')] *= 2
    elif health_objective == 'Eye Health':
        query_point[0, numeric_columns.index('Vitamin A')] *= 2
    elif health_objective == 'Joint Health':
        query_point[0, numeric_columns.index('Vitamin C')] *= 2
    elif health_objective == 'Brain Health':
        query_point[0, numeric_columns.index('Vitamin C')] *= 2
    elif health_objective == 'Muscle Gain':
        query_point[0, numeric_columns.index('Protein')] *= 2
    logger.info("Query point after adjustment for %s: %s", health_objective, query_point)
    return query_point

def recommend_foods(user_profile, main_df, knn_model, scaler):
    preferred_foods = get_preferred_foods(user_profile, main_df)
    preferred_features = preferred_foods.drop(columns=['Food and Serving'])

    if preferred_features.empty:
        return pd.DataFrame(columns=main_df.columns)  # Return an empty DataFrame with the same columns

    scaled_preferred_features = scaler.transform(preferred_features)
    logger.info("Scaled preferred features: %s", scaled_preferred_features)

    # Adjust the query point based on health objectives
    health_objective = user_profile.Health_Objectives
    query_point = scaled_preferred_features.mean(axis=0).reshape(1, -1)
    query_point = adjust_query_point(query_point, health_objective)

    distances, indices = knn_model.kneighbors(query_point)

    recommended_foods = preferred_foods.iloc[indices[0]]

    return recommended_foods

@app.post("/recommended")
def recommend(user_profile: UserProfile):
    try:
        logger.info("Received request: %s", user_profile)
        recommendations = recommend_foods(user_profile, nutritional_data, knn, scaler)
        logger.info("Recommendations: %s", recommendations)
        return {"recommended_foods": recommendations['Food and Serving'].tolist()}
    except Exception as e:
        logger.error("Error in recommendation: %s", str(e))
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
