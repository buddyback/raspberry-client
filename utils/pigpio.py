import asyncio
import time

import pigpio

from config.settings import VIBRATION_PIN


class PigpioClient:
    def __init__(self, host="localhost", port=8888):
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
            await asyncio.sleep(0.1)
            self.pi.write(VIBRATION_PIN, 0)
            await asyncio.sleep(0.5)
            self.pi.write(VIBRATION_PIN, 1)
            await asyncio.sleep(0.1)
            self.pi.write(VIBRATION_PIN, 0)
        finally:
            self._alert_running = False

    async def long_alert(self, intensity = 100):
        # map intensity from 0-100 to 150-255 (for PWM duty cycle)
        duty_cycle = int(150 + (intensity / 100) * 105)

        if not self.pi.connected or self._alert_running:
            return
        try:
            self._alert_running = True
            self.pi.set_PWM_dutycycle(VIBRATION_PIN, duty_cycle)
            await asyncio.sleep(1)
            self.pi.write(VIBRATION_PIN, 0)
            await asyncio.sleep(1)
            self.pi.set_PWM_dutycycle(VIBRATION_PIN, duty_cycle)
            await asyncio.sleep(1)
            self.pi.write(VIBRATION_PIN, 0)
            await asyncio.sleep(1)
            self.pi.set_PWM_dutycycle(VIBRATION_PIN, duty_cycle)
            await asyncio.sleep(1)
            self.pi.write(VIBRATION_PIN, 0)
        finally:
            self._alert_running = False


    def long_alert_thread(self, intensity = 100):
        # map intensity from 0-100 to 150-255 (for PWM duty cycle)
        duty_cycle = int(150 + (intensity / 100) * 105)

        if not self.pi.connected or self._alert_running:
            return
        try:
            self._alert_running = True
            self.pi.set_PWM_dutycycle(VIBRATION_PIN, duty_cycle)
            time.sleep(1)
            self.pi.write(VIBRATION_PIN, 0)
            time.sleep(1)
            self.pi.set_PWM_dutycycle(VIBRATION_PIN, duty_cycle)
            time.sleep(1)
            self.pi.write(VIBRATION_PIN, 0)
            time.sleep(1)
            self.pi.set_PWM_dutycycle(VIBRATION_PIN, duty_cycle)
            time.sleep(1)
            self.pi.write(VIBRATION_PIN, 0)
        finally:
            self._alert_running = False