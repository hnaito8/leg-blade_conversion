import cv2
import mediapipe as mp
import numpy as np

# MediaPipeのPoseモジュール
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# 描画用ユーティリティ
mp_drawing = mp.solutions.drawing_utils

# 義足画像を読み込む（投下情報も読み込むため -1）
prosthetic_img = cv2.imread(
    "./xiborg_nu.png",
    cv2.IMREAD_UNCHANGED,
)

# カメラ起動
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("カメラから映像を取得できませんでした")
        break

    # BGRをRGBに変換（MediaPipeはRGB想定）
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    # 骨格検出できたら
    if results.pose_landmarks:
        # # 骨格を描画する
        # mp_drawing.draw_landmarks(
        #     frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        # )

        landmarks = results.pose_landmarks.landmark

        # 右膝と右足首の座標を取得
        right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
        right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]

        # 画像サイズに合わせて座標をスケーリング
        h, w, _ = frame.shape
        knee_x, knee_y = int(right_knee.x * w), int(right_knee.y * h)
        ankle_x, ankle_y = int(right_ankle.x * w), int(right_ankle.y * h)

        # 確認用に、膝と足首に丸を描く（デバッグ用）
        cv2.circle(frame, (knee_x, knee_y), 10, (255, 0, 0), -1)
        cv2.circle(frame, (ankle_x, ankle_y), 10, (0, 255, 0), -1)

        # 膝から足首へのベクトル
        dx = ankle_x - knee_x
        dy = ankle_y - knee_y
        angle = np.degrees(np.arctan2(dy, dx)) - 90  # 縦方向基準に調整

        # 長さを測って、義足のリサイズサイズを決める
        length = int(np.sqrt(dx * dx + dy * dy))
        prosthetic_resized = cv2.resize(prosthetic_img, (length, length))
        # 横幅を少し細めにしたいなら
        # prosthetic_img, (int(length * 0.5), length)

        # 画像を回転させる
        (h_p, w_p) = prosthetic_resized.shape[:2]
        center = (w_p // 2, h_p // 2)
        rot_mat = cv2.getRotationMatrix2D(
            center, -angle, 1.0
        )  # 膝下と合わせるため、angleを-1した
        prosthetic_rotated = cv2.warpAffine(
            prosthetic_resized,
            rot_mat,
            (w_p, h_p),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )

        # 貼り付け位置を決める（膝の位置を基準）
        top_left_x = knee_x - w_p // 2
        top_left_y = knee_y

        # 透明部分を考慮して合成
        if 0 <= top_left_x < frame.shape[1] and 0 <= top_left_y < frame.shape[0]:
            # 貼り付ける範囲をフレームサイズに収める
            paste_w = min(w_p, frame.shape[1] - top_left_x)
            paste_h = min(h_p, frame.shape[0] - top_left_y)

            prosthetic_crop = prosthetic_rotated[0:paste_h, 0:paste_w]

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

    # 画面に表示
    cv2.imshow("Pose Detection", frame)

    #'q'キーを押すと終了
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
