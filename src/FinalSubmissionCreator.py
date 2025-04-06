# #############################################################################
# ##### Competition: DATATHON 2025 - ETH Zürich
# ##### Challenge: ALPIQ Challenge (Predict Energy Consumption for Italy and Spain)
# ##### Team: Gradient Descenters
# ##### Members: Kevin Riehl, Alexander Faroux, Cedric Zeiter, Anja Sjöström
# #############################################################################




# #############################################################################
# IMPORTS
# #############################################################################
import pandas as pd
import os


DEFAULT_MODEL = "gradboost"
country = "IT"
df_matcher = pd.read_excel("PriorityModelAssignment.xlsx", sheet_name=country)

df_complete = None
for idx, row in df_matcher.iterrows():
    customer = row["Customer"]
    modelType = row["modelType"]
    if str(customer)=="nan":
        continue
    if not os.path.exists("../data/forecasts/"+str(modelType)+"/"+customer+".csv"):
        modelType = DEFAULT_MODEL
    df = pd.read_csv("../data/forecasts/"+modelType+"/"+customer+".csv")
    df = df[["time", "consumption"]]
    df = df.rename(columns={"consumption": "VALUEMWHMETERINGDATA_"+customer})
    if df_complete is None:
        df_complete = df.copy()
    else:
        df_complete = df_complete.merge(df, on="time", how="left")
df_complete = df_complete.rename(columns={"time": "DATETIME"})
df_complete.to_csv("data/final_submission/final_submission_"+str(country)+".csv", index=None)
