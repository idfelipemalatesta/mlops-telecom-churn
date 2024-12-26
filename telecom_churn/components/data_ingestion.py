import os
import sys

import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split

from telecom_churn.entity.config_entity import DataIngestionConfig
from telecom_churn.entity.artifact_entity import DataIngestionArtifact
from telecom_churn.exception import TelecomChurnException
from telecom_churn.logger import logging
from telecom_churn.data_access.telecom_data import TelecomData


class DataIngestion:
    def __init__(self,data_ingestion_config:DataIngestionConfig=DataIngestionConfig()):
        """
        :param data_ingestion_config: configuration for data ingestion
        """
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise TelecomChurnException(e,sys)
        

    def export_data_into_feature_store(self)->DataFrame:
        """
        Method Name :   export_data_into_feature_store
        Description :   This method exports data from mongodb to csv file
        
        Output      :   data is returned as artifact of data ingestion components
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            logging.info(f"Arquivo: data_ingestion/export_data_into_feature_store = Exporting data from mongodb")

            telecom_data = TelecomData()
            dataframe = telecom_data.export_collection_as_dataframe(collection_name=self.data_ingestion_config.collection_name)

            logging.info(f"Arquivo: data_ingestion/export_data_into_feature_store = Shape of dataframe: {dataframe.shape}")

            feature_store_file_path  = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path,exist_ok=True)

            logging.info(f"Arquivo: data_ingestion/export_data_into_feature_store = Saving exported data into feature store file path: {feature_store_file_path}")

            dataframe.to_csv(feature_store_file_path,index=False,header=True)
            return dataframe

        except Exception as e:
            raise TelecomChurnException(e,sys)
        
     

    def split_data_as_train_test(self,dataframe: DataFrame) ->None:
        """
        Method Name :   split_data_as_train_test
        Description :   This method splits the dataframe into train set and test set based on split ratio 
        
        Output      :   Folder is created in s3 bucket
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Arquivo: data_ingestion/split_data_as_train_test = Entered split_data_as_train_test method of Data_Ingestion class")

        try:
            train_set, test_set = train_test_split(dataframe, test_size=self.data_ingestion_config.train_test_split_ratio)
            logging.info("Arquivo: data_ingestion/split_data_as_train_test = Performed train test split on the dataframe")
            logging.info("Arquivo: data_ingestion/split_data_as_train_test = Exited split_data_as_train_test method of Data_Ingestion class")

            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path,exist_ok=True)
            
            logging.info(f"Arquivo: data_ingestion/split_data_as_train_test = Exporting train and test file path.")
            train_set.to_csv(self.data_ingestion_config.training_file_path,index=False,header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path,index=False,header=True)

            logging.info(f"Arquivo: data_ingestion/split_data_as_train_test = Exported train and test file path.")
        except Exception as e:
            raise TelecomChurnException(e, sys) from e
        


    
    def initiate_data_ingestion(self) ->DataIngestionArtifact:
        """
        Method Name :   initiate_data_ingestion
        Description :   This method initiates the data ingestion components of training pipeline 
        
        Output      :   train set and test set are returned as the artifacts of data ingestion components
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Arquivo: data_ingestion/initiate_data_ingestion = Entered initiate_data_ingestion method of Data_Ingestion class")

        try:
            dataframe = self.export_data_into_feature_store()

            logging.info("Arquivo: data_ingestion/initiate_data_ingestion = Got the data from ARCHIVE")

            self.split_data_as_train_test(dataframe)

            logging.info("Arquivo: data_ingestion/initiate_data_ingestion = Performed train test split on the dataset")
            logging.info("Arquivo: data_ingestion/initiate_data_ingestion = Exited initiate_data_ingestion method of Data_Ingestion class")

            data_ingestion_artifact = DataIngestionArtifact(trained_file_path=self.data_ingestion_config.training_file_path,
            test_file_path=self.data_ingestion_config.testing_file_path)
            
            logging.info(f"Arquivo: data_ingestion/initiate_data_ingestion = Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise TelecomChurnException(e, sys) from e