import tensorflow as tf
import dill as pk
import os
import requests

base_dir = os.path.dirname(__file__)

class ModelLoader:
    _instance = None

    @staticmethod
    def get_instance():
        if ModelLoader._instance is None:
            ModelLoader._instance = ModelLoader()
        return ModelLoader._instance

    def __init__(self) -> None:
        if ModelLoader._instance is not None:
            raise Exception('This is a singleton')
        
        # Define model path
        model_path = os.path.join('/tmp', 'lstm_model_new.keras')  # Use /tmp to avoid read/write issues
        
        # Check if the model file exists, else download it
        if not os.path.exists(model_path):
            url = "https://drive.google.com/file/d/17bC6lE8HuAx618DhxkOoK7H9OBypfwhl/view?usp=drive_link"  # Replace with Google Drive/Dropbox direct download link
            response = requests.get(url, stream=True)
            with open(model_path, "wb") as f:
                f.write(response.content)

        # Load model
        self.model = tf.keras.models.load_model(model_path)

    def get_data(self):
        return self.model

class TokenizerLoader:
    _instance = None

    @staticmethod
    def get_instance():
        if TokenizerLoader._instance is None:
            TokenizerLoader._instance = TokenizerLoader()
        return TokenizerLoader._instance

    def __init__(self) -> None:
        if TokenizerLoader._instance is not None:
            raise Exception('This is a singleton')
        
        tokenizer_path = os.path.join(base_dir, 'tokenizer.pkl')
        
        with open(tokenizer_path, 'rb') as f:
            self.tokenizer = pk.load(f)

    def get_data(self):
        return self.tokenizer
