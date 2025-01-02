import cv2
import numpy as np

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