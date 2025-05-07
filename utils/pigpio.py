import time
import pigpio
from config.settings import VIBRATION_PIN
import subprocess
import asyncio


class PigpioClient:
    def __init__(self, host='localhost', port=8888):
        # loop = asyncio.get_event_loop()
        self.pi = pigpio.pi(host, port)
        self.pi.set_mode(VIBRATION_PIN, pigpio.OUTPUT)
        self._alert_running = False

    async def short_alert(self):
        if not self.pi.connected or self._alert_running:
            return
        try:
            self._alert_running = True
            self.pi.write(VIBRATION_PIN, 1)
            # Wait for 0.1 seconds
            #subprocess.Popen('echo "vibro 1"', shell=True)
            await asyncio.sleep(0.1)
            self.pi.write(VIBRATION_PIN, 0)
            #subprocess.Popen('echo "vibro 0"', shell=True)
            await asyncio.sleep(0.5)
            self.pi.write(VIBRATION_PIN, 1)
            #subprocess.Popen('echo "vibro 1"', shell=True)
            await asyncio.sleep(0.1)
            self.pi.write(VIBRATION_PIN, 0)
            #subprocess.Popen('echo "vibro 0"', shell=True)
        finally:
            self._alert_running = False