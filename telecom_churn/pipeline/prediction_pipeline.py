import os
import sys

import numpy as np
import pandas as pd
from telecom_churn.entity.config_entity import TelecomChurnPredictorConfig
from telecom_churn.entity.s3_estimator import TelecomChurnEstimator
from telecom_churn.exception import TelecomChurnException
from telecom_churn.logger import logging
from telecom_churn.utils.main_utils import read_yaml_file
from pandas import DataFrame


class TelecomChurnData:
    def __init__(self,
                SeniorCitizen,
                Partner,
                Dependents,
                tenure,
                MultipleLines,
                InternetService,
                OnlineSecurity,
                OnlineBackup,
                DeviceProtection,
                TechSupport,
                StreamingTV,
                StreamingMovies,
                Contract,
                PaperlessBilling,
                PaymentMethod,
                MonthlyCharges,
                TotalCharges
                ):
        """
        Usvisa Data constructor
        Input: all features of the trained model for prediction
        """
        try:
            self.SeniorCitizen = SeniorCitizen
            self.Partner = Partner
            self.Dependents = Dependents
            self.tenure = tenure
            self.MultipleLines = MultipleLines
            self.InternetService = InternetService
            self.OnlineSecurity = OnlineSecurity
            self.OnlineBackup = OnlineBackup
            self.DeviceProtection = DeviceProtection
            self.TechSupport = TechSupport
            self.StreamingTV = StreamingTV
            self.StreamingMovies = StreamingMovies
            self.Contract = Contract
            self.PaperlessBilling = PaperlessBilling
            self.PaymentMethod = PaymentMethod
            self.MonthlyCharges = MonthlyCharges
            self.TotalCharges = TotalCharges


        except Exception as e:
            raise TelecomChurnException(e, sys) from e

    def get_telecom_input_data_frame(self)-> DataFrame:
        """
        This function returns a DataFrame from TelecomChurnData class input
        """
        try:
            
            telecom_input_dict = self.get_telecom_data_as_dict()
            return DataFrame(telecom_input_dict)
        
        except Exception as e:
            raise TelecomChurnException(e, sys) from e


    def get_telecom_data_as_dict(self):
        """
        This function returns a dictionary from TelecomChurnData class input 
        """
        logging.info("Entered get_telecom_data_as_dict method as TelecomChurnData class")

        try:
            input_data = {
                "SeniorCitizen": [self.SeniorCitizen],
                "Partner": [self.Partner],
                "Dependents": [self.Dependents],
                "tenure": [self.tenure],
                "MultipleLines": [self.MultipleLines],
                "InternetService": [self.InternetService],
                "OnlineSecurity": [self.OnlineSecurity],
                "OnlineBackup": [self.OnlineBackup],
                "DeviceProtection": [self.DeviceProtection],
                "TechSupport": [self.TechSupport],
                "StreamingTV": [self.StreamingTV],
                "StreamingMovies": [self.StreamingMovies],
                "Contract": [self.Contract],
                "PaperlessBilling": [self.PaperlessBilling],
                "PaymentMethod": [self.PaymentMethod],
                "MonthlyCharges": [self.MonthlyCharges],
                "TotalCharges": [self.TotalCharges],
            }

            logging.info("Created telecom data dict")

            logging.info("Exited get_telecom_data_as_dict method as TelecomChurnData class")

            return input_data

        except Exception as e:
            raise TelecomChurnException(e, sys) from e

class TelecomChurnClassifier:
    def __init__(self,prediction_pipeline_config: TelecomChurnPredictorConfig = TelecomChurnPredictorConfig(),) -> None:
        """
        :param prediction_pipeline_config: Configuration for prediction the value
        """
        try:
            # self.schema_config = read_yaml_file(SCHEMA_FILE_PATH)
            self.prediction_pipeline_config = prediction_pipeline_config
        except Exception as e:
            raise TelecomChurnException(e, sys)


    def predict(self, dataframe) -> str:
        """
        This is the method of TelecomChurnClassifier
        Returns: Prediction in string format
        """
        try:
            logging.info("Entered predict method of TelecomChurnClassifier class")
            model = TelecomChurnEstimator(
                bucket_name=self.prediction_pipeline_config.model_bucket_name,
                model_path=self.prediction_pipeline_config.model_file_path,
            )
            result =  model.predict(dataframe)
            
            return result
        
        except Exception as e:
            raise TelecomChurnException(e, sys)