import requests
import yaml

class OllamaClient:
    def __init__(self, config_llm):
        self.api_url = config_llm["api_url"]
        self.model_name = config_llm["model_name"]
    
    def query(self, prompt):
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False 
        }

        try:
            print("Invio domanda: ")
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}

    def preprocessingQuery(self, prompt):
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }

        try:
            print("Riassunto in corso...")
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
