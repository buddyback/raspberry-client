from urllib.parse import urljoin

import requests
import aiohttp
import asyncio
from urllib.parse import urljoin

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

        self.last_sensitivity = -1
        self.last_session_status = -1
        self.last_vibration_intensity = -1

        self._polling_task = None

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

    async def poll(self, timeout=60):
        print("Polling...")
        url = urljoin(self.base_url, "/devices/settings/")
        params = {
            "last_sensitivity": self.last_sensitivity,
            "last_session_status": self.last_session_status,
            "last_vibration_intensity": self.last_vibration_intensity
        }

        try:
            async with aiohttp.ClientSession(headers=self.headers) as session:
                async with session.get(url, params=params, timeout=timeout) as response:
                    response.raise_for_status()
                    data = await response.json()
                    self.last_sensitivity = data["sensitivity"]
                    self.last_session_status = data["has_active_session"]
                    self.last_vibration_intensity = data["vibration_intensity"]
                    return data
        except aiohttp.ClientError as e:
            print(f"Error sending posture data: {e}")
            raise

    async def _poll_loop(self):
        print("Starting polling loop...")
        while True:
            await self.poll()
            await asyncio.sleep(60)

    def start_polling(self):
        if not self._polling_task:
            self._polling_task = asyncio.create_task(self._poll_loop())