from fastapi import FastAPI
from routers import serve, train



# model_loader = ModelLoader(mlflow_model_uri=)
app = FastAPI()

app.include_router(serve.router)
app.include_router(train.router)
