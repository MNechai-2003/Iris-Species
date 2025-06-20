import sys
import kaggle 
from pathlib import Path
from dotenv import load_dotenv
from abc import ABC, abstractmethod
from typing_extensions import override

sys.path.append("/home/nikolay/Deloitte/project_task/Iris/")
load_dotenv()


class Authenticator(ABC):
    @abstractmethod
    def authenticate(self):
        pass


class DataProvider(ABC):
    @abstractmethod
    def download_dataset(self, dataset_name: str) -> Path:
        pass


class KaggleAuthenticator(Authenticator):
    @override
    def authenticate(self):
        try:
            api = kaggle.KaggleApi()
            api.authenticate()
            return api
        except Exception as e:
            raise ConnectionError(f"Kaggle authentication failed: {e}")


class KaggleDataProvider(DataProvider):
    """Provider for loading Iris dataset through Kaggle API"""

    def __init__(self, download_path: str, authenticator: Authenticator) -> None:
        self.authenticator = authenticator
        self.download_path = Path(download_path)
        self.api = self.authenticator.authenticate()

    def download_dataset(self, dataset_name: str) -> Path:
        try:
            self.api.dataset_download_files(
                dataset=dataset_name,
                path=self.download_path,
                unzip=True
            )
            return Path(self.download_path / dataset_name.split("/")[-1])
        except Exception as e:
            raise ConnectionError(f"Kaggle dataset download failed: {str(e)}")


class DataProviderFactory:
    @staticmethod
    def create_kaggle_provider(download_path: Path) -> KaggleDataProvider:
        authenticator = KaggleAuthenticator()
        return KaggleDataProvider(
            download_path=download_path,
            authenticator=authenticator
        )


if __name__ == "__main__":
    try:
        config = {
            "dataset_name": "uciml/iris",
            "download_root": "/home/nikolay/Deloitte/project_task/Iris/src/core/datafiles/raw"
        }
        provider = DataProviderFactory.create_kaggle_provider(download_path=config['download_root'])
        dataset_path = provider.download_dataset(config["dataset_name"])
        print(f"Dataset successfully downloaded to: {dataset_path}")
    except Exception as e:
        print(f"Error: {str(e)}")
