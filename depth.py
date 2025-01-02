import cv2
import numpy as np
import matplotlib.pyplot as plt
from disparity import calcDisparityByZNCC



def main():
    image = cv2.imread('images/2000.jpg')

    image = cv2.resize(image, (352, 528))

    blue, _, red = cv2.split(image)

    # ===視差を算出する===
    disp = calcDisparityByZNCC(red, blue, 17, 21)
    fig, ax = plt.subplots()
    im = ax.imshow(disp, cmap='jet')
    fig.colorbar(im, ax=ax)
    plt.show()

    # d_map = np.zeros(std_img.shape, dtype=np.float32)
    depth_map = np.zeros(disp.shape, dtype=np.float32)

    v = 463188.704432995
    u = 68.9522453187647
    w = 0
    a = -42.9284

    v_neg = 455756.211142968
    u_neg = 90.4626116573327
    w_neg = 0
    a_neg = -43.5302

    depth_list = []


    # ピクセル単位の計算
    for y in range(0, disp.shape[0]):
        for x in range(0, disp.shape[1]):
            # エッジ条件を仮にコメントアウト（必要に応じて処理を追加）
            d = disp[y, x]
            if d != 0:
                if d > 0 and abs(d) <= 5:
                    z = v / (float(y) - a*d - u) + w
                    depth_map[y, x] = z
                    depth_list.append(z)
                elif d < 0 and abs(d) <= 5:
                    z = v_neg / (float(y) - a_neg*d - u_neg) + w_neg
                    depth_map[y, x] = z
                    depth_list.append(z)
    #show histogram
    plt.hist(depth_list, bins=100, range=(0, 3200))
    plt.show()
    
    fig, ax = plt.subplots()
    im = ax.imshow(depth_map, cmap='jet')
    fig.colorbar(im, ax=ax)
    plt.show()

if __name__ == '__main__':
    main()