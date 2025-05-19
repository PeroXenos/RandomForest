from fastapi import FastAPI
import joblib

def add_feature_names_endpoint(app: FastAPI):
    @app.get("/feature_names")
    async def get_feature_names():
        try:
            feature_names = joblib.load('feature_names.joblib')
            return list(feature_names)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

# Add this to the main app
add_feature_names_endpoint(app)