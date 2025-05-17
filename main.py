import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import os

# 警告を抑制
import warnings
warnings.filterwarnings('ignore')

# MNISTデータセットの読み込み
def load_mnist_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # 正規化
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    return (x_train, y_train), (x_test, y_test)

def get_template_image():
    print("Preparing template images...")
    # 各数字（0-9）のテンプレート画像を準備
    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    templates = [[] for _ in range(10)]
    for i in range(len(x_train)):
        digit = y_train[i]
        if len(templates[digit]) < 10:  # 各数字につき10枚だけテンプレートとして使用
            templates[digit].append(x_train[i])
        
        # 全ての数字について少なくとも10枚のテンプレートが集まったら終了
        if all(len(t) > 10 for t in templates):
            break
    print("Template image preparation completed.")
    return templates

def save_template_images(templates):
    print("Saving template images...")
    fig, axes = plt.subplots(10, 10, figsize=(20, 20))
    for i in range(10):
        for j in range(10):
            axes[i][j].imshow(templates[i][j], cmap='gray')
            axes[i][j].axis('off')
    fig.tight_layout()
    save_path = os.path.join('results', 'template_images.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    print("Template images saved.")
    return True

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def template_matching(test_image, template_image):
    test_img = (test_image * 255).astype(np.uint8)
    template = (template_image * 255).astype(np.uint8)
    
    result = cv2.matchTemplate(test_img, template, cv2.TM_CCOEFF_NORMED)
    _, score, _, _ = cv2.minMaxLoc(result)
    
    return score

def set_font():
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['MS Gothic']
    plt.rcParams['font.serif'] = ['MS Gothic']

# 結果を3つのサンプルを一枚の画像に保存する関数
def save_combined_results(test_images, target1, target2, score1, score2, path):
    set_font()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    i = 0
    axes[i].imshow(test_images, cmap='gray')
    # 日本語タイトルを設定
    title_text = f'リファレンス'
    axes[i].set_title(title_text)
    axes[i].axis('off')
    
    i = 1
    axes[i].imshow(target1, cmap='gray')
    # 日本語タイトルを設定
    title_text = f'ターゲット1, スコア: {score1:.4f}'
    axes[i].set_title(title_text)
    axes[i].axis('off')
    
    i = 2
    axes[i].imshow(target2, cmap='gray')
    # 日本語タイトルを設定
    title_text = f'ターゲット2, スコア: {score2:.4f}'
    axes[i].set_title(title_text)
    axes[i].axis('off')
    
    plt.tight_layout()
    
    save_path = os.path.join('results', f'{path}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return save_path

def main0(template_images):
    '''
    0と異なる種類の0のテンプレート画像を比較する
    '''
    print("Executing template matching main1...")
    test_image = template_images[0][0]
    target1 = test_image
    
    score1 = template_matching(test_image, target1)
    print(f'Score1: {score1:.4f}')
    
    set_font()
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    i = 0
    axes[i].imshow(test_image, cmap='gray')
    axes[i].set_title('リファレンス')
    axes[i].axis('off')
    
    i = 1
    axes[i].imshow(target1, cmap='gray')
    axes[i].set_title(f'ターゲット, スコア: {score1:.4f}')
    axes[i].axis('off')
    
    plt.tight_layout()
    
    save_path = os.path.join('results', f'template_matching0.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Result image saved to: {save_path}')
    
def main1(template_images):
    '''
    0と異なる種類の0のテンプレート画像を比較する
    '''
    print("Executing template matching main1...")
    test_image = template_images[0][0]
    target1 = test_image
    target2 = template_images[0][7]
    
    score1 = template_matching(test_image, target1)
    score2 = template_matching(test_image, target2)
    print(f'Score1: {score1:.4f}')
    print(f'Score2: {score2:.4f}')
    
    save_path = save_combined_results(test_image, target1, target2, score1, score2, 'template_matching1')
    print(f'Result image saved to: {save_path}')
    
def main2(template_images):
    '''
    0と異なる種類の0(似ている)のテンプレート画像を比較する
    '''
    print("Executing template matching main2...")
    test_image = template_images[0][0]
    target1 = test_image
    target2 = template_images[0][2]
    
    score1 = template_matching(test_image, target1)
    score2 = template_matching(test_image, target2)
    print(f'Score1: {score1:.4f}')
    print(f'Score2: {score2:.4f}')
    
    save_path = save_combined_results(test_image, target1, target2, score1, score2, 'template_matching2')
    print(f'Result image saved to: {save_path}')

def main3(template_images):
    '''
    0と異なる数字のテンプレート画像を比較する
    '''
    print("Executing template matching main3...")
    test_image = template_images[0][0]
    target1 = test_image
    target2 = template_images[1][0]
    
    score1 = template_matching(test_image, target1)
    score2 = template_matching(test_image, target2)
    print(f'Score1: {score1:.4f}')
    print(f'Score2: {score2:.4f}')
    
    save_path = save_combined_results(test_image, target1, target2, score1, score2, 'template_matching3')
    print(f'Result image saved to: {save_path}')
    
def main4(template_images):
    '''
    回転した画像との比較
    '''
    print("Executing template matching main4...")
    degrees = 30
    test_image = template_images[0][0]
    target1 = test_image
    target2 = rotate_image(test_image, degrees)
    
    score1 = template_matching(test_image, target1)
    score2 = template_matching(test_image, target2)
    print(f'Score1: {score1:.4f}')
    print(f'Score2: {score2:.4f}')
    
    save_path = save_combined_results(test_image, target1, target2, score1, score2, 'template_matching4')
    print(f'Result image saved to: {save_path}')
    
def main5(template_images):
    '''
    縮小した画像との比較
    '''
    print("Executing template matching main5...")
    scale = 0.5
    test_image = template_images[0][0]
    target1 = test_image
    target2 = cv2.resize(test_image, (0, 0), fx=scale, fy=scale)
    
    score1 = template_matching(test_image, target1)
    score2 = template_matching(test_image, target2)
    print(f'Score1: {score1:.4f}')
    print(f'Score2: {score2:.4f}')
    
    save_path = save_combined_results(test_image, target1, target2, score1, score2, 'template_matching5')
    print(f'Result image saved to: {save_path}')
    
def main6(template_images):
    '''
    拡大した画像との比較
    '''
    print("Executing template matching main6...")
    scale = 3.0
    test_image = template_images[0][0]
    target1 = test_image
    target2 = cv2.resize(test_image, (0, 0), fx=scale, fy=scale)
    
    score1 = template_matching(test_image, target1)
    score2 = template_matching(test_image, target2)
    print(f'Score1: {score1:.4f}')
    print(f'Score2: {score2:.4f}')
    
    save_path = save_combined_results(test_image, target1, target2, score1, score2, 'template_matching6')
    print(f'Result image saved to: {save_path}')

if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)
    
    template_images = get_template_image()
    
    save_template_images(template_images)
    main0(template_images)
    main1(template_images)
    main2(template_images)
    main3(template_images)
    main4(template_images)
    main5(template_images)
    main6(template_images)
    print("Processing completed.")