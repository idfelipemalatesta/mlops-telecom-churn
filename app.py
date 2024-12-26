from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import HTMLResponse, RedirectResponse
from uvicorn import run as app_run

from typing import Optional

from telecom_churn.constants import APP_HOST, APP_PORT
from telecom_churn.pipeline.prediction_pipeline import TelecomChurnData, TelecomChurnClassifier
from telecom_churn.pipeline.training_pipeline import TrainPipeline

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory='templates')

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DataForm:
    def __init__(self, request: Request):
        self.request: Request = request
        self.SeniorCitizen : Optional[str] = None
        self.Partner: Optional[str] = None
        self.Dependents: Optional[str] = None
        self.tenure: Optional[str] = None
        self.MultipleLines: Optional[str] = None
        self.InternetService: Optional[str] = None
        self.OnlineSecurity: Optional[str] = None
        self.OnlineBackup: Optional[str] = None
        self.DeviceProtection: Optional[str] = None
        self.TechSupport: Optional[str] = None
        self.StreamingTV: Optional[str] = None
        self.StreamingMovies : Optional[str] = None
        self.Contract: Optional[str] = None
        self.PaperlessBilling: Optional[str] = None
        self.PaymentMethod: Optional[str] = None
        self.MonthlyCharges: Optional[str] = None
        self.TotalCharges: Optional[str] = None
        

    async def get_telecom_data(self):
        form = await self.request.form()
        self.SeniorCitizen = form.get("SeniorCitizen")
        self.Partner = form.get("Partner")
        self.Dependents = form.get("Dependents")
        self.tenure = form.get("tenure")
        self.MultipleLines = form.get("MultipleLines")
        self.InternetService = form.get("InternetService")
        self.OnlineSecurity = form.get("OnlineSecurity")
        self.OnlineBackup = form.get("OnlineBackup")
        self.DeviceProtection = form.get("DeviceProtection")
        self.TechSupport = form.get("TechSupport")
        self.StreamingTV = form.get("StreamingTV")
        self.StreamingMovies = form.get("StreamingMovies")
        self.Contract = form.get("Contract")
        self.PaperlessBilling = form.get("PaperlessBilling")
        self.PaymentMethod = form.get("PaymentMethod")
        self.MonthlyCharges = form.get("MonthlyCharges")
        self.TotalCharges = form.get("TotalCharges")

@app.get("/", tags=["authentication"])
async def index(request: Request):

    return templates.TemplateResponse(
            "telecom.html",{"request": request, "context": "Rendering"})


@app.get("/train")
async def trainRouteClient():
    try:
        train_pipeline = TrainPipeline()

        train_pipeline.run_pipeline()

        return Response("Training successful !!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")


@app.post("/")
async def predictRouteClient(request: Request):
    try:
        form = DataForm(request)
        await form.get_telecom_data()
        
        telecom_data = TelecomChurnData(
                                SeniorCitizen= form.SeniorCitizen,
                                Partner = form.Partner,
                                Dependents = form.Dependents,
                                tenure = form.tenure,
                                MultipleLines= form.MultipleLines,
                                InternetService= form.InternetService,
                                OnlineSecurity = form.OnlineSecurity,
                                OnlineBackup= form.OnlineBackup,
                                DeviceProtection= form.DeviceProtection,
                                TechSupport= form.TechSupport,
                                StreamingTV = form.StreamingTV,
                                StreamingMovies= form.StreamingMovies,
                                Contract= form.Contract,
                                PaperlessBilling = form.PaperlessBilling,
                                PaymentMethod= form.PaymentMethod,
                                MonthlyCharges= form.MonthlyCharges,
                                TotalCharges= form.TotalCharges,
                                )
        
        telecom_df = telecom_data.get_telecom_input_data_frame()
        print(telecom_df)

        model_predictor = TelecomChurnClassifier()

        value = model_predictor.predict(dataframe=telecom_df)[0]

        status = None
        if value == 1:
            status = "CHURN"
        else:
            status = "NO CHURN"

        return templates.TemplateResponse(
            "telecom.html",
            {"request": request, "context": status},
        )
        
    except Exception as e:
        return {"status": False, "error": f"{e}"}


if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)