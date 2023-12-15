from utils import util

def cabai_rawit_merah(num_prediction):
    return util(dataset_name="Cabai_Rawit_Merah", model_name="Cabai Rawit Merah-GRU_1.0", num_prediction=num_prediction)

# def cabai_rawit_merah(num_prediction):
#     return util(dataset_name="Cabai_Merah_Keriting", model_name="cabai-rawit-merah", num_prediction=num_prediction)

def cabai_merah_keriting(num_prediction):
    return util(dataset_name="Cabai_Merah_Keriting", model_name="Cabai_Merah_Keriting-LSTM_2.0", num_prediction=num_prediction)