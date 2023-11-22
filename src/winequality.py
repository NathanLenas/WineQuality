from fastapi import FastAPI
from enum import Enum
from typing import Optional
from pydantic import BaseModel
import pandas as pd
import modele as m


class Wine(BaseModel):
    id: Optional[int] = -1
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    ph: float
    sulphates: float
    alcohol: float
    quality: Optional[int] = -1

app = FastAPI()



@app.get("/api/model/")
def getSerializedModel():
    return {"model" : m.getModel().to_json()}
    


@app.get('/api/model/description')
def getModelInfos():
    return {"model" : [{"parameters" : [{"name": "id", "type": "int"}, {"name": "fixed_acidity", "type": "float"}, {"name": "volatile_acidity", "type": "float"}, {"name": "citric_acid", "type": "float"}, {"name": "residual_sugar", "type": "float"}, {"name": "chlorides", "type": "float"}, {"name": "free_sulfur_dioxide", "type": "float"}, {"name": "total_sulfur_dioxide", "type": "float"}, {"name": "density", "type": "float"}, {"name": "pH", "type": "float"}, {"name": "sulphates", "type": "float"}, {"name": "alcohol", "type": "float"}, {"name": "quality", "type": "int"} ]
                        }], "summary": [m.description()] }

@app.put('/api/model/')
def addEntry(wine: Wine):
    data = pd.read_csv("../data/Wines.csv")
    wine.id = data['Id'].max() + 1
    if wine.quality == -1:
        return {"error": "quality is missing"}
    next_row = pd.DataFrame({"fixed acidity": wine.fixed_acidity, "volatile acidity": wine.volatile_acidity, "citric acid": wine.citric_acid, "residual sugar": wine.residual_sugar, "chlorides": wine.chlorides, "free sulfur dioxide": wine.free_sulfur_dioxide, "total sulfur dioxide": wine.total_sulfur_dioxide, "density": wine.density, "pH": wine.ph, "sulphates": wine.sulphates, "alcohol": wine.alcohol, "quality": wine.quality, "Id": wine.id}, index=[0])
    
    newdata = pd.concat([data, next_row], axis=0).reset_index(drop=True)
    
    newdata.to_csv("../data/Wines.csv", index=False)
    
    return {"id": wine.id}
# Example body for post request : {"id": 1, "fixed_acidity": 7.4, "volatile_acidity": 0.7, "citric_acid": 0, "residual_sugar": 1.9, "chlorides": 0.076, "free_sulfur_dioxide": 11, "total_sulfur_dioxide": 34, "density": 0.9978, "ph": 3.51, "sulphates": 0.56, "alcohol": 9.4, "quality": 5}

# addEntry(Wine(id=-1, fixed_acidity=7.4, volatile_acidity=0.7, citric_acid=0, residual_sugar=1.9, chlorides=0.076, free_sulfur_dioxide=11, total_sulfur_dioxide=34, density=0.9978, pH=3.51, sulphates=0.56, alcohol=9.4, quality=5))

@app.post('/api/model/retrain')
def retrainModel():
    m.train()
    return {"status": "ok"}

@app.post('/api/model/predict')
def predict(wine: Wine):
    return {"quality" : m.predire(wine.fixed_acidity, wine.volatile_acidity, wine.citric_acid, wine.residual_sugar, wine.chlorides, wine.free_sulfur_dioxide, wine.total_sulfur_dioxide, wine.density, wine.ph, wine.sulphates, wine.alcohol)}

@app.get("/api/model/predict")
def getVinParfait():
    p = m.parfait()
    
    wine_array = p 

    wine = Wine(
        fixed_acidity=wine_array[0],
        volatile_acidity=wine_array[1],
        citric_acid=wine_array[2],
        residual_sugar=wine_array[3],
        chlorides=wine_array[4],
        free_sulfur_dioxide=wine_array[5],
        total_sulfur_dioxide=wine_array[6],
        density=wine_array[7],
        ph=wine_array[8],
        sulphates=wine_array[9],
        alcohol=wine_array[10],
    )
    
    return wine
