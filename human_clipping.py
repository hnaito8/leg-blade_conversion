import cv2
import mediapipe as mp
import numpy as np

# MediaPipe セグメンテーションの初期化
mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# カメラ起動（0は内蔵カメラ）
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 画像をRGBに変換
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # セグメンテーションを実行
    results = selfie_segmentation.process(frame_rgb)

    # マスクを取得（人物部分だけをTrueに）
    mask = results.segmentation_mask
    condition = mask > 0.5  # 信頼度が50%以上の部分を人物と判定

    # 背景を黒に置き換える
    bg_image = np.zeros(frame.shape, dtype=np.uint8)

    # 人物だけを合成
    output_image = np.where(condition[:, :, None], frame, bg_image)

    # 表示
    cv2.imshow("Real-time Person Segmentation", output_image)

    # ESCキーで終了
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
