import cv2
import numpy as np
import math

import cv2
import numpy as np

import matplotlib.pyplot as plt

class EdgeDirection:
    BOTH = "both"
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"

def detectEdge(src, dtype, direction=EdgeDirection.BOTH, th_value=0):
    # カーネル定義
    kernel_x = np.array([
        [0, -1, 0],
        [0,  0, 0],
        [0,  1, 0]
    ], dtype=np.float64)

    kernel_y = np.array([
        [0,  0, 0],
        [-1, 0, 1],
        [0,  0, 0]
    ], dtype=np.float64)

    # フィルタ適用
    Gx = cv2.filter2D(src, cv2.CV_64F, kernel_x)
    Gy = cv2.filter2D(src, cv2.CV_64F, kernel_y)

    # エッジ方向に応じた処理
    if direction == EdgeDirection.BOTH:
        Gx2 = np.power(Gx, 2)
        Gy2 = np.power(Gy, 2)
        dst = np.sqrt(Gx2 + Gy2)
    elif direction == EdgeDirection.HORIZONTAL:
        dst = np.abs(Gx)
    elif direction == EdgeDirection.VERTICAL:
        dst = np.abs(Gy)
    else:
        raise ValueError("Invalid edge direction specified")

    # 型変換
    dst = dst.astype(dtype)
    
    # 閾値処理
    _, dst = cv2.threshold(dst, th_value, 255, cv2.THRESH_TOZERO)

    return dst

def calcDisparityByZNCC(std_img, tmp_img, kernel_size, search_range_x,th_value=35):
    if len(std_img.shape) != 2 or len(tmp_img.shape) != 2:
        raise ValueError("入力画像は1チャンネルの画像を入力してください")
    
    k_size = (kernel_size - 1) // 2
    s_range = (search_range_x - 1) // 2

    d_map = np.zeros(std_img.shape, dtype=np.float32)
    edge = detectEdge(std_img, np.uint8, direction=EdgeDirection.VERTICAL, th_value=th_value)

    # エッジ検出 (仮にソーベルのみを使う)
    edge_std_img = cv2.Sobel(std_img, cv2.CV_32F, 1, 0)
    edge_tmp_img = cv2.Sobel(tmp_img, cv2.CV_32F, 1, 0)

    edge_std_img = cv2.convertScaleAbs(edge_std_img)
    edge_tmp_img = cv2.convertScaleAbs(edge_tmp_img)

    # ピクセル単位の計算
    for y in range(k_size, std_img.shape[0] - k_size):
        for x in range(s_range + k_size + 1, std_img.shape[1] - s_range - k_size - 1):
            # エッジ条件を仮にコメントアウト（必要に応じて処理を追加）
            if edge[y, x] > 0:
            
                # 中央のROI取得
                roi_r = edge_std_img[y - k_size:y + k_size + 1, x - k_size:x + k_size + 1]

                max_sim = -math.inf
                max_x = 0
                sim_array = np.zeros(search_range_x, dtype=np.float32)

                for shift_x in range(search_range_x):
                    shifted_x = x + s_range - k_size - shift_x
                    roi_b = edge_tmp_img[y - k_size:y + k_size + 1, shifted_x:shifted_x + kernel_size]

                    match_result = cv2.matchTemplate(roi_r, roi_b, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(match_result)

                    sim_array[shift_x] = max_val
                    if max_val > max_sim and shift_x != 0 and shift_x != search_range_x - 1:
                        max_sim = max_val
                        max_x = shift_x

                # サブピクセル精度の計算
                Rneg1 = sim_array[max_x - 1]
                R = sim_array[max_x]
                Rpos1 = sim_array[max_x + 1]
                subpixel = 0.0

                denominator = 4 * R - 2 * Rneg1 - 2 * Rpos1
                if denominator != 0:
                    subpixel = (Rpos1 - Rneg1) / denominator

                submin = max_x + subpixel
                abs_submin = submin - s_range
                d_map[y, x] = abs_submin

    return d_map