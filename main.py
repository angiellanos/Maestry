from fastapi import FastAPI
from proyecto.main_router import main__router

app = FastAPI()

app.include_router(main__router)

@app.get("/")
def read_root():
    return {"Proyecto SURA"}
