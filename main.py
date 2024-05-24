import pathlib
import shutil
from fastapi import FastAPI
from fastapi import FastAPI, File, UploadFile
import os

from core.ANN import ANNPredictor
from core.LR import LRPredictor
from core.SVM import SVMPredictor

app = FastAPI() 

@app.get("/") 
async def main_route(): 

  return {"message": "server is live"}



UPLOAD_FOLDER = "data"


os.makedirs(UPLOAD_FOLDER,exist_ok=True)

@app.post("/uploadfile/")
async def create_upload_file(svm:bool,lr:bool,ann:bool ,file: UploadFile = File(...)):
    
    file_location = os.path.join(UPLOAD_FOLDER, file.filename)
    
    with open(file_location, "wb") as file_object:
        file_object.write(await file.read())
    
 
    if svm:
      svm_test_in_path = f'data/{file.filename}'
      svm_predictions_output_path = r'output/svm_predictions.xlsx'
      svm_graph_output_path = r'output/svm_predictions.png'
      svm_model_path = r'model/svm_model.pkl'
      svm_scaler_path = r'model/svm_scaler.pkl'
    # Initialize and run the SVMPredictor
      svm_predictor = SVMPredictor(svm_test_in_path, svm_predictions_output_path, svm_graph_output_path, svm_model_path, svm_scaler_path)
      svm_predictor.run()
    else:
      svm_predictions_output_path = None
      svm_graph_output_path = None

    if lr:
      lr_test_in_path = f'data/{file.filename}'
      lr_predictions_output_path = r'output/lr_predictions.xlsx'
      lr_graph_output_path = r'output/lr_predictions.png'
      lr_model_path = r'model/linear_model.pkl'
      lr_scaler_path = r'model/scaler.pkl'

      lr_predictor = LRPredictor(lr_test_in_path, lr_predictions_output_path, lr_graph_output_path, lr_model_path, lr_scaler_path)
      lr_predictor.run()
    else:
      lr_predictions_output_path = None
      lr_graph_output_path = None
       


    if ann:
      ann_test_in_path = f'data/{file.filename}'
      ann_predictions_output_path = r'output/ann_predictions.xlsx'
      ann_graph_output_path = r'output/ann_predictions.png'
      ann_model_path = r'model/ann_model.keras'
      ann_scaler_input_path = r'model/ann_scaler_input.pkl'
      ann_scaler_output_path = r'model/ann_scaler_output.pkl'

      ann_predictor = ANNPredictor(ann_test_in_path, ann_predictions_output_path, ann_graph_output_path, ann_model_path, ann_scaler_input_path, ann_scaler_output_path)
      ann_predictor.run()
    else:
      ann_predictions_output_path = None
      ann_graph_output_path = None


    responce = [
      {
      "ann_predictions": {
        "img": pathlib.Path(ann_graph_output_path).resolve().as_uri() ,
        "excel": pathlib.Path(ann_predictions_output_path).resolve().as_uri() 
      }},
      {
      "lr_predictions": {
        "img":  pathlib.Path(lr_graph_output_path).resolve().as_uri() ,
        "excel": pathlib.Path(lr_predictions_output_path).resolve().as_uri() ,
      }},
      {
      "svm_predictions": {
        "img": pathlib.Path(svm_graph_output_path).resolve().as_uri() ,
        "excel": pathlib.Path(svm_predictions_output_path).resolve().as_uri() ,
      }}
      ]



    return responce

