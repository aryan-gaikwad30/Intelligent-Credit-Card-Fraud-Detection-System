from fastapi import FastAPI

app = FastAPI(title="Intelligent Credit Card Fraud Detection System API")

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Backend scaffold is running!"}
