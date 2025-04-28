from urllib.parse import urljoin

import requests

from config.settings import BODY_COMPONENTS


class HttpClient:
    def __init__(self, base_url, api_key, device_id):
        self.base_url = base_url
        self.api_key = api_key
        self.device_id = device_id
        self.headers = {
            "X-API-Key": self.api_key,
            "X-Device-ID": self.device_id,
        }

    @staticmethod
    def _serialize_posture(raw_data):
        components = []
        for component_name, attributes in BODY_COMPONENTS.items():
            if attributes["parameter"] in raw_data:
                components.append(
                    {
                        "component_type": component_name,
                        "is_correct": raw_data[attributes["parameter"]] < attributes["reference"],
                        "score": int(raw_data[attributes["parameter"]] / attributes["reference"] * 100),
                        "correction": raw_data["issues"].get(component_name, "No issues detected"),
                    }
                )
        return {"components": components}

    def send_posture_data(self, analyzer_results):
        endpoint = "/posture-data/"
        request_body = self._serialize_posture(analyzer_results)
        full_url = urljoin(self.base_url, endpoint)
        try:
            response = requests.post(full_url, json=request_body, headers=self.headers, timeout=10)
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            print(f"Error sending posture data: {e}")
            raise
