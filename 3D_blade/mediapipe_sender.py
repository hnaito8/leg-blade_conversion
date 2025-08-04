# mediapipe_sender.py
import cv2
import mediapipe as mp
import asyncio
import websockets
import json

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)
cap = cv2.VideoCapture(0)


async def send_pose(websocket):
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        if results.pose_world_landmarks:
            landmarks = results.pose_world_landmarks.landmark
            knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
            ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]

            data = {
                "knee": {"x": knee.x, "y": knee.y, "z": knee.z},
                "ankle": {"x": ankle.x, "y": ankle.y, "z": ankle.z},
            }
            await websocket.send(json.dumps(data))

        await asyncio.sleep(0.03)  # 30 FPS


async def main():
    async with websockets.serve(send_pose, "localhost", 8765):
        print("WebSocket送信中 : ws://localhost:8765")
        await asyncio.Future()  # run forever


asyncio.run(main())
