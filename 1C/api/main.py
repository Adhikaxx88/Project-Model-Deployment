from fastapi import FastAPI

from api.routers import predict

app = FastAPI(title="Credit Score Predictor API")

app.include_router(predict.router)


@app.get("/")
def root():
    return {"status": "ok"}
