# File: main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ML.Routes import predict, backtest # Assuming your routers are in an 'app' subfolder

# Initialize the FastAPI app
app = FastAPI(
    title="Trading ML Service",
    description="An API to serve trading model predictions and run backtests.",
    version="1.0.0"
)

# --- Middleware ---
# Set up CORS to allow requests from your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Or be more specific, e.g., ["https://your-frontend.vercel.app"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Routers ---
# Include the routers from your other files
app.include_router(predict.router, prefix="/api", tags=["Predictions"])
app.include_router(backtest.router, prefix="/api", tags=["Backtesting"])


# --- Root Endpoint ---
@app.get("/", tags=["Health Check"])
def root():
    return {"status": "ok", "message": "Welcome to the Trading ML Service!"}
