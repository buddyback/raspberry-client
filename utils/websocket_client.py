import asyncio
import json
import sys
import time

# Client heartbeat interval (in seconds)
HEARTBEAT_INTERVAL = 30


# TODO: Move the websocket logic to posture detector and use this only for sending data
class WebSocketClient:
    def __init__(self, base_url: str, device_id: str, api_key: str):
        self.device_id = device_id
        self.api_key = api_key
        self.base_url = base_url
        self.uri = f"{self.base_url}/ws/device-connection/{self.device_id}/?api_key={self.api_key}"
        print(self.uri)
        self.websocket = None
        self.heartbeat_running = True

    async def wait_responses(self):
        """
        Test the device settings WebSocket connection.
        Usage: python device_settings_websocket_test.py <device_id> <api_key>
        """

        print(f"Connecting to: {self.uri}")
        print(f"Current time: {time.strftime('%H:%M:%S')}")

        previous_settings = {}
        while True:
            update = await self.websocket.recv()
            # msg_counter += 1
            # current_time = time.strftime("%H:%M:%S")
            # print(f"\n[{current_time}] Message #{msg_counter} received: {update}")

            try:
                data = json.loads(update)

                # Handle different message types
                if data.get("type") == "heartbeat_ack":
                    print("❤️ Heartbeat acknowledged by server")

                elif data.get("type") == "session_status":
                    # Handle session status events
                    action = data.get("action")
                    last_session_status = data.get("last_session_status", False)

                    if action == "start_session":
                        print(f"🟢 SESSION STARTED - Active: {last_session_status}")
                        # Your device could start monitoring posture here

                    elif action == "stop_session":
                        print(f"🔴 SESSION ENDED - Active: {last_session_status}")
                        # Your device could stop monitoring posture here

                elif data.get("type") == "posture_data_response":
                    # Handle response to our posture data submission
                    status = data.get("status")
                    if status == "success":
                        print("✅ Posture data successfully saved")
                    else:
                        print(f"❌ Error saving posture data: {data.get('error')}")

                elif data.get("type") == "settings":
                    # Extract the actual settings data
                    settings_data = data.get("data", {})

                    # Update session status from settings if available
                    if "last_session_status" in settings_data:
                        last_session_status = settings_data["last_session_status"]

                    # Highlight changes in settings
                    changes = []
                    for key, value in settings_data.items():
                        if key not in previous_settings or previous_settings[key] != value:
                            changes.append(f"{key}: {previous_settings.get(key, 'N/A')} → {value}")

                    if changes:
                        print("\n" + "!" * 50)
                        print("🔄 DEVICE SETTINGS UPDATED:")
                        for change in changes:
                            print(f"  ✓ {change}")
                        print("!" * 50)
                    else:
                        print(f"Settings received (no changes): {settings_data}")

                    # Update previous settings
                    previous_settings = settings_data.copy()

                else:
                    print(f"Unknown message type: {data}")

            except Exception as e:
                print(f"Error processing message: {str(e)}")
            finally:
                pass
                # Stop heartbeat task
                # heartbeat_running = False
                # if 'heartbeat_task' in locals() and heartbeat_task:
                #     heartbeat_task.cancel()
                #
                # # Stop user command task
                # if 'user_task' in locals() and user_task:
                #     user_task.cancel()

    async def send_heartbeats(self):
        """Periodically send heartbeats to the server"""
        while self.heartbeat_running:
            try:
                # Sleep for the interval
                await asyncio.sleep(HEARTBEAT_INTERVAL)

                # Send heartbeat
                print(f"❤️ SENDING HEARTBEAT at {time.strftime('%H:%M:%S')}")
                await self.websocket.send(json.dumps({"type": "heartbeat"}))

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error sending heartbeat: {str(e)}")
                break

    async def get_settings(self):
        # First, receive the initial settings
        response = await self.websocket.recv()
        response_object = json.loads(response)
        settings = None

        if response_object.get("type") == "settings":
            # Extract the actual settings data
            if response_object.get("data"):
                settings = response_object.get("data", {})

        return settings
        # Check if we already have an active session
        # if isinstance(initial_data, dict) and initial_data.get("data"):
        #     last_session_status = initial_data.get("data", {}).get("has_active_session", False)
        # elif isinstance(initial_data, dict):
        #     last_session_status = initial_data.get("has_active_session", False)

    async def send_posture_data(self, results):
        """Send a single posture reading with randomized scores"""
        # Generate somewhat realistic scores (weighted towards medium-good posture)

        # Create posture data payload
        posture_data = {
            "type": "posture_data",
            "data": {
                "components": [
                    {"component_type": "neck", "score": int(results.get("neck_score"))},
                    {"component_type": "torso", "score": int(results.get("torso_score"))},
                    {"component_type": "shoulders", "score": int(results.get("shoulders_score"))},
                ]
            },
        }

        # Send data and print what we're sending
        # print(f"\n📤 Sending posture data: neck={neck_score}, torso={torso_score}, shoulders={shoulders_score}")
        print(f"\n📤 Sending posture data: {posture_data}")
        await self.websocket.send(json.dumps(posture_data))

    async def process_user_commands(self):
        """Process user commands from stdin while WebSocket is running"""
        while True:
            # Read command from stdin
            line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
            command = line.strip().lower()

            if command == "data":
                # Send a single posture data sample
                await self.send_single_posture_reading(self.websocket)
            elif command == "heartbeat":
                # Send a manual heartbeat
                print("❤️ Sending manual heartbeat")
                await self.websocket.send(json.dumps({"type": "heartbeat"}))
