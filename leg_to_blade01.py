# 背景を選べるやつはどうか？右手をあげて背景変更

import cv2
import mediapipe as mp
import numpy as np
import time

# MediaPipe Pose モジュール
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# 描画ユーティリティ
mp_drawing = mp.solutions.drawing_utils

# 義足画像（透過付きPNG）を読み込み
prosthetic_img = cv2.imread("./xiborg_nu.png", cv2.IMREAD_UNCHANGED)

# カメラ起動
cap = cv2.VideoCapture(0)

# === 背景をキャプチャ ===
print("背景をキャプチャします。3秒間静止してください...")
time.sleep(3)
ret, background = cap.read()
if not ret:
    print("背景画像の取得に失敗しました。終了します。")
    cap.release()
    exit()

print("背景キャプチャ完了。処理開始！")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("カメラから映像を取得できませんでした")
        break

    # BGR→RGB（MediaPipeはRGB前提）
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # 右膝・右足首の座標取得
        right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
        right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]

        h, w, _ = frame.shape
        knee_x, knee_y = int(right_knee.x * w), int(right_knee.y * h)
        ankle_x, ankle_y = int(right_ankle.x * w), int(right_ankle.y * h)

        # デバッグ用マーカー
        cv2.circle(frame, (knee_x, knee_y), 10, (255, 0, 0), -1)
        cv2.circle(frame, (ankle_x, ankle_y), 10, (0, 255, 0), -1)

        # 膝～足首の矩形を背景で上書き（消す）
        x1 = max(0, min(knee_x, ankle_x) - 10)
        y1 = max(0, min(knee_y, ankle_y) - 10)
        x2 = min(w, max(knee_x, ankle_x) + 10)
        y2 = min(h, max(knee_y, ankle_y) + 10)
        frame[y1:y2, x1:x2] = background[y1:y2, x1:x2]

        # 義足のベクトルとサイズ
        dx = ankle_x - knee_x
        dy = ankle_y - knee_y
        angle = np.degrees(np.arctan2(dy, dx)) - 90
        length = int(np.sqrt(dx * dx + dy * dy))

        # 義足画像をリサイズ
        prosthetic_resized = cv2.resize(prosthetic_img, (length, length))

        # 回転
        h_p, w_p = prosthetic_resized.shape[:2]
        center = (w_p // 2, h_p // 2)
        rot_mat = cv2.getRotationMatrix2D(center, -angle, 1.0)
        prosthetic_rotated = cv2.warpAffine(
            prosthetic_resized,
            rot_mat,
            (w_p, h_p),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )

        # 膝に義足を貼り付ける位置
        top_left_x = knee_x - w_p // 2
        top_left_y = knee_y

        # 合成処理（透明部分を考慮）
        if 0 <= top_left_x < w and 0 <= top_left_y < h:
            paste_w = min(w_p, w - top_left_x)
            paste_h = min(h_p, h - top_left_y)
            prosthetic_crop = prosthetic_rotated[0:paste_h, 0:paste_w]

            # 透過チャネルを使った合成
            alpha = prosthetic_crop[:, :, 3] / 255.0
            for c in range(3):
                frame[
                    top_left_y : top_left_y + paste_h,
                    top_left_x : top_left_x + paste_w,
                    c,
                ] = (
                    alpha * prosthetic_crop[:, :, c]
                    + (1 - alpha)
                    * frame[
                        top_left_y : top_left_y + paste_h,
                        top_left_x : top_left_x + paste_w,
                        c,
                    ]
                )

    # 表示
    cv2.imshow("Pose Detection", frame)

    # 'q'キーで終了
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
