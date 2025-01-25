import mlflow
import torch

class ModelLoader:
    def __init__(self, mlflow_model_uri: str):
        """
        Initializes the model loader with the MLflow model URI.

        Args:
            mlflow_model_uri (str): URI of the MLflow model to load.
        """
        self.mlflow_model_uri = mlflow_model_uri
        self.model = None
        self.tokenizer = None

    def get_model(self):
        if self.model is None or self.tokenizer is None:
            print(f"Loading model from MLflow server: {self.mlflow_model_uri}...")
            # Load the model using MLflow
            loaded_model = mlflow.pyfunc.load_model(self.mlflow_model_uri)
            self.model = loaded_model._model_impl.python_model.model  # Access underlying Hugging Face model
            self.tokenizer = loaded_model._model_impl.python_model.tokenizer
            self.model.to("cuda" if torch.cuda.is_available() else "cpu")
            print("Model loaded successfully from MLflow.")
        return self.model, self.tokenizer


if __name__ == "__main__":
    # import mlflow

    # mlflow.set_tracking_uri("http://mlflow:5000")
    # model_uri = "models:/my_model/production"
    # model_loader = ModelLoader(model_uri)
    # model, tokenizer = model_loader.get_model()

    import mlflow

    logged_model = 'runs:/9acee285129b429ba9243103cac82ff4/checkpoint-6750'

    # Load model as a PyFuncModel.
    loaded_model = mlflow.pyfunc.load_model(logged_model)

    # Predict on a Pandas DataFrame.
    import pandas as pd



    loaded_model.predict(pd.DataFrame(data))
    print(model)
