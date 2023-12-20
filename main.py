from fastapi import FastAPI
from utils import util_dua, get_commodities
from commodities import cabai_rawit_merah, cabai_merah_keriting, beras_medium
from typing import List

app = FastAPI()

@app.get("/")
def read_root():
    return 'TaniLink Machine Learning API'

@app.get("/commodities", response_model=List[dict])
def commodities():
    return get_commodities()

@app.get("/predictions/{commodity_id}/{num_prediction}")
def predict(commodity_id: str, num_prediction: int):
    return util_dua(commodity_id=commodity_id, num_prediction=num_prediction)

@app.get("/cabai-rawit-merah/{num_prediction}")
def cabai_rawit_merah_prediction(num_prediction: int):
    return cabai_rawit_merah(num_prediction=num_prediction)

@app.get("/cabai-merah-keriting/{num_prediction}")
def cabai_merah_keriting_prediction(num_prediction: int):
    return cabai_merah_keriting(num_prediction=num_prediction)

@app.get("/beras-medium/{num_prediction}")
def beras_medium_prediction(num_prediction: int):
    return beras_medium(num_prediction=num_prediction)

