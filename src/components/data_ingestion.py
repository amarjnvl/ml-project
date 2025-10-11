import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split 
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

# The dataclass decorator is used to define a class as a dataclass, which allows for the creation of immutable objects with a simple syntax. 
# It is essentially a wrapper around the __init__ method of the class, which is responsible for initializing the objects attributes. 
# When the dataclass decorator is used, the __init__ method is automatically generated from the classes attributes and is used to initialize the objects attributes. This makes it easier to define immutable objects with a simple syntax.
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save the dataframe to a csv file at the specified path, with headers and without index.
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info('Train test split initiated')

            # test_size: proportion of the dataset to include in the test set
            # random_state: seed used to shuffle the data before splitting it into train and test sets
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.error(f"Error occurred during data ingestion: {e}")
            raise CustomException(e, sys) from e
        
# The following code block is used to execute the code when this file is run as the main program (i.e., not imported as a module in another program).
# It is a common pattern in Python to have code that is only executed when the file is run directly (not imported), and this code block is used to protect that code from being executed when the file is imported as a module in another program.
if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)