import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from template_matching import get_template_image, save_template_images, save_combined_results, rotate_image

def shift_image(image, dx_orig, dy_orig, scale): # dx_orig, dy_orig は外部指定のオフセット
    rows, cols = image.shape
    
    # スケール後のオブジェクトを中央に配置するための追加のオフセット
    # 元の中心: (cols / 2, rows / 2)
    # スケール後のオブジェクトの中心 (もし左上に配置された場合): (cols * scale / 2, rows * scale / 2)
    adj_dx = (cols - cols * scale) / 2
    adj_dy = (rows - rows * scale) / 2
    
    # 外部指定のオフセットとセンタリングオフセットを合算
    final_dx = dx_orig + adj_dx
    final_dy = dy_orig + adj_dy
    
    M = np.float32([[scale, 0, final_dx], [0, scale, final_dy]])
    return cv2.warpAffine(image, M, (cols, rows))

def shift_matching(test_image, target_image, min_match_count=10, detector_type="sift"):
    target_sift_ready = (target_image * 255).astype(np.uint8)
    test_sift_ready = (test_image * 255).astype(np.uint8)

    # 1. 検出器オブジェクトの作成
    detector = None
    if detector_type.lower() == "sift":
        detector = cv2.SIFT_create()
    elif detector_type.lower() == "orb":
        detector = cv2.ORB_create(nfeatures=1000)
    elif detector_type.lower() == "akaze":
        detector = cv2.AKAZE_create()
    else:
        print(f"警告: 未知の検出器タイプ '{detector_type}'。デフォルトでSIFTを使用します。")
        detector = cv2.SIFT_create()

    # 2. 特徴点と記述子の計算
    kp1, des1 = detector.detectAndCompute(target_sift_ready, None)
    kp2, des2 = detector.detectAndCompute(test_sift_ready, None)

    if des1 is None or des2 is None or len(kp1) < min_match_count or len(kp2) < min_match_count:
        # print(f"{detector_type.upper()}: 特徴点が見つからないか、少なすぎます。")
        return 0.0

    # 3. 特徴点のマッチング
    bf = None
    if detector_type.lower() in ["orb", "akaze"]:
        # ORB/AKAZE はバイナリ記述子なので NORM_HAMMING を使用
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    else: # SIFT は float 記述子なので NORM_L2 を使用
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    
    all_matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    if all_matches:
        # k=2 で knnMatch を使った場合、各 match_pair は2つのマッチを含む可能性がある
        for match_pair in all_matches:
            if len(match_pair) == 2: # 2つのマッチがあることを確認
                m, n = match_pair
                if m.distance < 0.75 * n.distance: # Lowe's ratio test
                    good_matches.append(m)
            elif len(match_pair) == 1 and detector_type.lower() in ["orb", "akaze"]: 
                # BFMatcherでcrossCheck=Trueの場合や、k=1の場合、またはフィルタリングで1つになる場合
                # ただし、Lowe's ratio testのためにはk=2が必要
                # ここではk=2を前提としているので、この分岐は通常通らない
                # good_matches.append(match_pair[0]) 
                pass


    if len(good_matches) >= min_match_count:
        return float(len(good_matches))
    else:
        return 0.0

def draw_rect(image, x, y, w, h):
    ret = image.copy()
    cv2.rectangle(ret, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return ret

def main(template_images):
    print("Executing template matching main...")
    scale = 0.5
    test_image = template_images[0][0] 
    
    target1 = test_image # 元の画像
    image_for_transform = test_image
    
    target2_scaled = shift_image(image_for_transform, 0, 0, scale)
    target2 = rotate_image(target2_scaled, 30)
    
    detectors_to_test = ["sift", "orb", "akaze"]
    
    for detector_name in detectors_to_test:
        print(f"\n--- Testing with {detector_name.upper()} ---")
        
        # test_image と target1 (変形なし) でのマッチング
        score1 = shift_matching(test_image, target1, detector_type=detector_name)
        
        # test_image と target2 (変形あり) でのマッチング
        score2 = shift_matching(test_image, target2, detector_type=detector_name)
        
        print(f'Score1 ({detector_name.upper()} vs Original): {score1:.2f}')
        print(f'Score2 ({detector_name.upper()} vs Transformed): {score2:.2f}')

        save_filename = f'shift_{detector_name.lower()}_result'
        save_path = save_combined_results(test_image, target1, target2, score1, score2, save_filename)
        print(f'{detector_name.upper()} result image saved to: {save_path}')


if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)
    template_images = get_template_image()
    main(template_images)
    print("Processing completed.")