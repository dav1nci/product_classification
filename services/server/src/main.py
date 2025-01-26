from fastapi import FastAPI
from routers import serve, train

app = FastAPI()

app.include_router(serve.router)
app.include_router(train.router)
