import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# --- 1. データ準備関連の関数 ---
def load_mnist_data():
    """MNISTデータセットをロードして返す。"""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    print(f"MNISTデータ読み込み完了: 訓練データ {x_train.shape}, テストデータ {x_test.shape}")
    return (x_train, y_train), (x_test, y_test)

def create_transformed_query_image(template_image, angle, scale, tx, ty, bg_size_factor=1.5):
    """
    テンプレート画像に幾何学的変換を加えてクエリ画像を生成する。
    """
    rows, cols = template_image.shape
    center_x, center_y = cols / 2.0, rows / 2.0

    M = cv2.getRotationMatrix2D((center_x, center_y), angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty

    bg_h = int(rows * bg_size_factor)
    bg_w = int(cols * bg_size_factor)
    
    transformed_image = cv2.warpAffine(template_image, M, (bg_w, bg_h), borderValue=0)
    return transformed_image

# --- 2. SIFT処理関連の関数 ---
def extract_sift_features(image, image_name_for_debug="Image", 
                          contrast_threshold=0.04, 
                          edge_threshold=10, 
                          sigma=1.6):
    """
    画像からSIFTキーポイントとディスクリプタを抽出する。
    """
    if image is None:
        print(f"エラー: {image_name_for_debug} の画像がNoneです。")
        return None, None

    sift = cv2.SIFT_create(contrastThreshold=contrast_threshold,
                           edgeThreshold=edge_threshold,
                           sigma=sigma)
    keypoints, descriptors = sift.detectAndCompute(image, None)

    if descriptors is None or len(keypoints) == 0:
        print(f"エラー: {image_name_for_debug} からディスクリプタを抽出できませんでした。キーポイント数: {len(keypoints) if keypoints is not None else 0}")
        img_kp_viz = cv2.drawKeypoints(image, keypoints, None, color=(255,0,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        plt.figure(figsize=(5,5))
        plt.imshow(img_kp_viz, cmap='gray' if len(img_kp_viz.shape)==2 else None)
        plt.title(f"{image_name_for_debug} Keypoints (C:{contrast_threshold}, E:{edge_threshold}, S:{sigma})")
        plt.axis('off')
        plt.show(block=False)
        return None, None
    print(f"情報: {image_name_for_debug} から {len(keypoints)} 個のキーポイントを検出 (C:{contrast_threshold}, E:{edge_threshold}, S:{sigma})")
    return keypoints, descriptors

def match_sift_features(descriptors1, descriptors2, ratio_thresh=0.75): # ratio_thresh はここでデフォルト指定
    """
    2つのディスクリプタセットをマッチングし、良いマッチを選別する (Lowe's Ratio Test)。
    """
    if descriptors1 is None or descriptors2 is None:
        print("警告: マッチングのためのディスクリプタが一方または両方ありません。")
        return []

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    all_matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = []
    if all_matches: 
        for m_pair in all_matches:
            if len(m_pair) == 2: 
                m, n = m_pair
                if m.distance < ratio_thresh * n.distance:
                    good_matches.append(m)
            elif len(m_pair) == 1: 
                good_matches.append(m_pair[0]) 
    return good_matches

def compute_homography_from_matches(good_matches, kp1, kp2):
    """
    良いマッチからホモグラフィ行列を計算する。
    """
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    homography_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return homography_matrix

# --- 3. 描画・表示関連の関数 (修正箇所) ---
def draw_aspect_corrected_rectangle(image_to_draw_on, transformed_corners, original_template_shape, object_name="Object"):
    """
    変換後のコーナーから中心・角度・サイズを推定し、元のテンプレートのアスペクト比を
    維持した矩形を描画する。描画後の画像を返す。
    """
    if transformed_corners is None or len(transformed_corners) < 4:
        print(f"警告: {object_name} のための変換後コーナー座標が無効。矩形は描画されません。")
        return image_to_draw_on.copy()

    reshaped_corners = transformed_corners.reshape(-1, 2).astype(np.float32)
    if reshaped_corners.shape[0] < 4:
        print(f"警告: {object_name} のためのコーナー座標が4点未満。矩形は描画されません。")
        return image_to_draw_on.copy()
        
    # 1. 変換されたコーナーから、まず最小外接回転矩形を取得 (中心と角度の推定のため)
    fitted_rect = cv2.minAreaRect(reshaped_corners)
    center = fitted_rect[0]  # (cx, cy)
    angle_deg = fitted_rect[2] # 角度 (度)
    fitted_width = fitted_rect[1][0]
    fitted_height = fitted_rect[1][1]

    if fitted_width < 1 or fitted_height < 1: # 適合した矩形が小さすぎる場合
        print(f"警告: {object_name} の適合矩形が非常に小さいか無効です。サイズ: ({fitted_width:.2f}, {fitted_height:.2f})。元の最小外接矩形を描画します。")
        box_fallback = cv2.boxPoints(fitted_rect)
        box_fallback = np.intp(box_fallback)
        img_with_fallback_box = image_to_draw_on.copy()
        cv2.drawContours(img_with_fallback_box, [box_fallback], 0, (255, 165, 0), 1, cv2.LINE_AA) # オレンジ色でフォールバック
        return img_with_fallback_box

    # 2. 元のテンプレートのアスペクト比を取得
    template_h_orig, template_w_orig = original_template_shape[:2]
    if template_h_orig == 0: # ゼロ除算を避ける
        print(f"警告: {object_name} の元のテンプレートの高さが0です。")
        return image_to_draw_on.copy()
    template_aspect_ratio = template_w_orig / template_h_orig

    # 3. 新しい矩形の寸法を計算 (面積をfitted_rectの面積に合わせ、アスペクト比をテンプレートに合わせる)
    area_fitted = fitted_width * fitted_height
    
    new_h = np.sqrt(area_fitted / template_aspect_ratio)
    new_w = new_h * template_aspect_ratio

    if np.isnan(new_w) or np.isnan(new_h) or new_w < 1 or new_h < 1:
        print(f"警告: {object_name} のアスペクト比補正後のサイズ計算で問題。W:{new_w}, H:{new_h}。元の最小外接矩形を描画します。")
        box_fallback = cv2.boxPoints(fitted_rect)
        box_fallback = np.intp(box_fallback)
        img_with_fallback_box = image_to_draw_on.copy()
        cv2.drawContours(img_with_fallback_box, [box_fallback], 0, (255, 165, 0), 1, cv2.LINE_AA) # オレンジ色
        return img_with_fallback_box

    # 4. 新しいパラメータで回転矩形を定義し、その頂点を取得
    new_rect_params = (center, (float(new_w), float(new_h)), angle_deg)
    box_new = cv2.boxPoints(new_rect_params)
    box_new = np.intp(box_new)

    img_with_box = image_to_draw_on.copy()
    cv2.drawContours(img_with_box, [box_new], 0, (0, 0, 255), 1, cv2.LINE_AA) # 赤色でアスペクト比補正矩形
    center_pt = tuple(np.intp(center))
    cv2.circle(img_with_box, center_pt, 2, (255, 0, 0), -1) # 青色の中心点
    print(f"{object_name} - アスペクト比補正矩形: 中心 {center_pt}, 新サイズ ({new_w:.2f}, {new_h:.2f}), 角度 {angle_deg:.2f}")
    
    return img_with_box

def visualize_sift_matches(img_template, kp_template, img_query_with_box, kp_query, good_matches, title, save_path):
    """
    SIFTマッチング結果と検出されたオブジェクトの矩形を含む画像を表示する。
    """
    if len(img_template.shape) == 2: 
        img_template_color = cv2.cvtColor(img_template, cv2.COLOR_GRAY2BGR)
    else:
        img_template_color = img_template.copy()
    
    img_matches_viz = cv2.drawMatches(img_template_color, kp_template, img_query_with_box, kp_query,
                                      good_matches, None,
                                      matchColor=(0, 255, 255), 
                                      singlePointColor=None, 
                                      flags=cv2.DrawMatchesFlags_DEFAULT)

    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(img_matches_viz, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.savefig(save_path)

# --- 4. メインワークフロー関数 ---
def perform_sift_template_matching(template_img_gray, query_img_gray, object_name="Object",
                                   min_match_count=4,
                                   scale_factor=1.0,
                                   sift_contrast_thresh=0.04, 
                                   sift_edge_thresh=10,      
                                   sift_sigma=1.6,
                                   ratio_test_thresh=0.75): # ratio_test_thresh を追加
    """
    SIFTテンプレートマッチングの全プロセスを実行する。
    """
    print(f"\n--- \"{object_name}\" のマッチング処理開始 ---")
    print(f"使用するSIFTパラメータ: C={sift_contrast_thresh}, E={sift_edge_thresh}, S={sift_sigma}")
    print(f"画像拡大率: {scale_factor}, 最小マッチ数: {min_match_count}, Ratio Test閾値: {ratio_test_thresh}")

    scaled_template = template_img_gray
    scaled_query = query_img_gray
    if scale_factor > 1.0 and scale_factor != 1:
        scaled_template = cv2.resize(template_img_gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
        scaled_query = cv2.resize(query_img_gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

    kp_template, des_template = extract_sift_features(scaled_template, f"Template ({object_name})",
                                                      contrast_threshold=sift_contrast_thresh,
                                                      edge_threshold=sift_edge_thresh,
                                                      sigma=sift_sigma)
    if des_template is None: return

    kp_query, des_query = extract_sift_features(scaled_query, f"Query ({object_name})",
                                                contrast_threshold=sift_contrast_thresh,
                                                edge_threshold=sift_edge_thresh,
                                                sigma=sift_sigma)
    if des_query is None: return

    good_matches = match_sift_features(des_template, des_query, ratio_thresh=ratio_test_thresh) # ratio_thresh を渡す
    print(f"{object_name} - 良いマッチング数: {len(good_matches)}")

    img_query_color_for_drawing = cv2.cvtColor(query_img_gray, cv2.COLOR_GRAY2BGR)
    img_query_with_box = img_query_color_for_drawing.copy()
    homography_matrix = None
    dst_corners_scaled = None

    if len(good_matches) >= min_match_count:
        homography_matrix = compute_homography_from_matches(good_matches, kp_template, kp_query)
        if homography_matrix is not None:
            h_template_scaled, w_template_scaled = scaled_template.shape[:2]
            pts_template_corners_scaled = np.float32([
                [0, 0], [0, h_template_scaled - 1],
                [w_template_scaled - 1, h_template_scaled - 1], [w_template_scaled - 1, 0]
            ]).reshape(-1, 1, 2)
            dst_corners_scaled = cv2.perspectiveTransform(pts_template_corners_scaled, homography_matrix)
            
            if dst_corners_scaled is not None:
                dst_corners_original_scale = dst_corners_scaled / scale_factor
                # 修正: 新しい描画関数を呼び出す
                img_query_with_box = draw_aspect_corrected_rectangle(
                                            img_query_color_for_drawing, 
                                            dst_corners_original_scale, 
                                            template_img_gray.shape, # 元のテンプレートの形状を渡す
                                            object_name)
                # print(f"{object_name} - アスペクト比補正矩形を描画しました (元のスケール)。") # draw_aspect_corrected_rectangle内で表示
            else:
                print(f"{object_name} - 座標変換(perspectiveTransform)に失敗。矩形は描画されません。")
        else:
            print(f"{object_name} - ホモグラフィを計算できませんでした。矩形は描画されません。")
    else:
        print(f"{object_name} - 十分なマッチング ({len(good_matches)}/{min_match_count}) が見つかりませんでした。矩形は描画されません。")

    scaled_query_color_for_match_viz = cv2.cvtColor(scaled_query, cv2.COLOR_GRAY2BGR)
    if homography_matrix is not None and dst_corners_scaled is not None:
        # 拡大画像上にアスペクト比補正矩形を描画 (可視化用)
        scaled_query_color_for_match_viz = draw_aspect_corrected_rectangle(
                                                scaled_query_color_for_match_viz,
                                                dst_corners_scaled,
                                                scaled_template.shape, # スケール後のテンプレート形状
                                                f"{object_name} (on scaled query)")

    save_path = os.path.join("results", f"SIFT_matching_{object_name}.png")
    visualize_sift_matches(scaled_template, kp_template,
                           scaled_query_color_for_match_viz, kp_query, good_matches,
                           title=f'SIFT Matches for "{object_name}" (on scaled images)',
                           save_path=save_path)
    
    if not np.array_equal(img_query_with_box, img_query_color_for_drawing):
        plt.figure(figsize=(15,5))
        plt.subplot(1,3,1)
        plt.imshow(template_img_gray, cmap='gray')
        plt.title(f"Template")
        plt.axis('off')
        plt.subplot(1,3,2)
        plt.imshow(cv2.cvtColor(img_query_color_for_drawing, cv2.COLOR_BGR2RGB))
        plt.title(f"Original Scale Query")
        plt.axis('off')
        plt.subplot(1,3,3)
        plt.imshow(cv2.cvtColor(img_query_with_box, cv2.COLOR_BGR2RGB))
        plt.title(f"Detected '{object_name}' on Original Scale Query (Aspect Corrected)")
        plt.axis('off')
        save_path = os.path.join("results", f"SIFT_{object_name}.png")
        plt.savefig(save_path)
    print(f"--- \"{object_name}\" のマッチング処理終了 ---\n")

# ユーザーが提供したmain_flow関数
def main_flow(x_train_data, y_train_data, digit_to_process, ratio_thresh_val=0.75): # ratio_thresh_valを追加
    template_indices = np.where(y_train_data == digit_to_process)[0]

    template_image = x_train_data[template_indices[0]]
    query_image_original = create_transformed_query_image(template_image,
                                                          angle=20.0, scale=0.7,
                                                          tx=10, ty=8, bg_size_factor=2.0)
    print(f"テンプレート画像 (数字 '{digit_to_process}', サイズ: {template_image.shape}) と "
          f"クエリ画像 (サイズ: {query_image_original.shape}) を準備しました。")

    perform_sift_template_matching(template_image, query_image_original,
                                   object_name=f"MNIST Digit {digit_to_process}",
                                   min_match_count=4,      
                                   scale_factor=2.0,       
                                   sift_contrast_thresh=0.01, 
                                   sift_edge_thresh=25,       
                                   sift_sigma=1.6,
                                   ratio_test_thresh=ratio_thresh_val) # ratio_threshを渡す

# --- メイン実行ブロック ---
if __name__ == '__main__':
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

    (x_train, y_train), _ = load_mnist_data()

    # Ratio Test 閾値 0.65 で実行
    print("\n\n=== Ratio Test 閾値 = 0.65 ===")
    main_flow(x_train, y_train, 5, ratio_thresh_val=0.65)
    main_flow(x_train, y_train, 7, ratio_thresh_val=0.65) # クエリは '5' の変形

    # # Ratio Test 閾値 0.7 で実行 (より厳しく)
    # print("\n\n=== Ratio Test 閾値 = 0.7 ===")
    # main_flow(x_train, y_train, 5, ratio_thresh_val=0.7)
    # main_flow(x_train, y_train, 7, ratio_thresh_val=0.7) # クエリは '5' の変形

    plt.show()