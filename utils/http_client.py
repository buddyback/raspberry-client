import asyncio
import math
from urllib.parse import urljoin

import aiohttp
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

        self.last_sensitivity = -1
        self.last_session_status = False
        self.last_vibration_intensity = -1

        self._polling_task = None
        self._session = None

    def _serialize_posture(self, raw_data):
        components = []
        for component_name, attributes in BODY_COMPONENTS.items():
            if attributes["parameter"] in raw_data:

                score = int(raw_data[attributes["score"]])

                components.append(
                    {
                        "component_type": component_name,
                        "is_correct": score >= attributes["default_threshold"] if self.last_sensitivity == -1 else self.last_sensitivity,
                        "score": score,
                        "correction": raw_data["issues"].get(component_name, "No issues detected"),
                    }
                )
        return {"components": components}

    def send_posture_data(self, analyzer_results):
        endpoint = "/posture-data/"
        request_body = self._serialize_posture(analyzer_results)
        print(request_body)
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
            "last_session_status": str(self.last_session_status),
            "last_vibration_intensity": self.last_vibration_intensity,
        }

        try:
            if not self._session:
                self._session = aiohttp.ClientSession(headers=self.headers)
            async with self._session.get(url, params=params, timeout=timeout) as response:
                response.raise_for_status()  # Verifica che non ci siano errori
                data = await response.json()  # Ottieni i dati dalla risposta
                self.last_sensitivity = data["sensitivity"]
                self.last_session_status = data["has_active_session"]
                self.last_vibration_intensity = data["vibration_intensity"]
                return data
        except aiohttp.ClientError as e:
            print(f"Error polling settings: {e}")
            raise

    async def _poll_loop(self):
        print("Starting long polling loop...")
        while True:
            try:
                await self.poll()
            except Exception as e:
                print(f"Polling failed: {e}")
                await asyncio.sleep(1)

    def start_polling(self):
        if not self._polling_task:
            self._polling_task = asyncio.create_task(self._poll_loop())

    async def cleanup(self):
        if self._session:
            await self._session.close()
            self._session = None
        if not self._polling_task:
            self._polling_task = asyncio.create_task(self._poll_loop())
