import time

import pigpio
from config.settings import VIBRATION_PIN

class PigpioClient:
    def __init__(self, host='localhost', port=8888):
        self.pi = pigpio.pi(host, port)
        self.pi.set_mode(VIBRATION_PIN, pigpio.OUTPUT)

    async def short_alert(self, asyncio=None):
        if not self.pi.connected:
            return
        self.pi.write(VIBRATION_PIN, 1)
        # Wait for 0.1 seconds
        await asyncio.sleep(0.1)
        self.pi.write(VIBRATION_PIN, 0)
        await asyncio.sleep(0.5)
        self.pi.write(VIBRATION_PIN, 1)
        await asyncio.sleep(0.1)
        self.pi.write(VIBRATION_PIN, 0)