from fastapi import FastAPI
from commodities import cabai_rawit_merah, cabai_merah_keriting, beras_medium

app = FastAPI()

@app.get("/")
def read_root():
    return 'TaniLink Machine Learning API'

@app.get("/cabai-rawit-merah/{num_prediction}")
def cabai_rawit_merah_prediction(num_prediction: int):
    return cabai_rawit_merah(num_prediction=num_prediction)

@app.get("/cabai-merah-keriting/{num_prediction}")
def cabai_merah_keriting_prediction(num_prediction: int):
    return cabai_merah_keriting(num_prediction=num_prediction)

@app.get("/beras-medium/{num_prediction}")
def beras_medium_prediction(num_prediction: int):
    return beras_medium(num_prediction=num_prediction)
