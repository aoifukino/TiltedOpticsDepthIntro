import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from disparity import calcDisparityByZNCC



def main():
    correct_depth = 1934
    image = cv2.imread(f'images/test/{correct_depth}.jpg')

    image = cv2.resize(image, (352, 528))

    blue, _, red = cv2.split(image)

    # ===視差を算出する===
    disp = calcDisparityByZNCC(red, blue, 17, 21, 20)
    disp[disp > 5] = 0
    disp[disp < -5] = 0
    fig, ax = plt.subplots()
    ax.pcolormesh(disp, cmap='jet', norm=Normalize(vmin=-5, vmax=5))
    im = ax.imshow(disp, cmap='jet')
    fig.colorbar(im, ax=ax)
    plt.show()

    # d_map = np.zeros(std_img.shape, dtype=np.float32)
    depth_map = np.zeros(disp.shape, dtype=np.float32)
    error_map = np.zeros(disp.shape, dtype=np.float32)

    v = 463188.704432995
    u = 68.9522453187647
    w = 0
    a = -42.9284

    v_neg = 455756.211142968
    u_neg = 90.4626116573327
    w_neg = 0
    a_neg = -43.5302

    depth_list = []
    error_list = []


    # ピクセル単位の計算
    for y in range(0, disp.shape[0]):
        for x in range(0, disp.shape[1]):
            d = disp[y, x]
            if d != 0:
                if d > 0 and abs(d) <= 5:
                    z = v / (float(y) - a*d - u) + w
                    if z > 0 and z < 3000:
                        error = abs(z - correct_depth)
                        depth_map[y, x] = z
                        error_map[y, x] = error
                        depth_list.append(z)
                        error_list.append(error)
                    # print(f"z: {z}, error: {error}")
                elif d < 0 and abs(d) <= 5:
                    z = v_neg / (float(y) - a_neg*d - u_neg) + w_neg
                    if z > 0 and z < 3000:
                        error = abs(z - correct_depth)
                        depth_map[y, x] = z
                        error_map[y, x] = error
                        depth_list.append(z)
                        error_list.append(error)
                        # print(f"z: {z}, error: {error}")

    # ===平均距離値と絶対平均誤差を表示===
    mean_depth = np.mean(depth_list)
    mean_error = np.mean(error_list)
    # max error and index
    max_error = np.max(error_list)
    max_error_index = error_list.index(max_error)
    print(f"最大誤差（mm）: {max_error}, index: {max_error_index}")
    print(f"平均推定距離値（mm）: {mean_depth}")
    print(f"絶対平均誤差（mm）: {mean_error}")

    # ===ヒストグラムを表示する===
    plt.hist(depth_list, bins=100, range=(0, 3200))
    plt.xlabel("Depth (mm)")
    plt.show()

    # ===距離マップと誤差マップを表示する===
    fig, ax = plt.subplots()
    fig2, ax2 = plt.subplots()
    ax.pcolormesh(depth_map, cmap='jet', norm=Normalize(vmin=0, vmax=3000))
    im = ax.imshow(depth_map,cmap='jet',vmin=0, vmax=3000)
    im2 = ax2.imshow(error_map, cmap='jet',vmin=0, vmax=200)
    ax.set_title('Depth Map')
    ax2.set_title('Error Map')
    fig.colorbar(im, ax=ax)
    fig2.colorbar(im2, ax=ax2)
    plt.show()


if __name__ == '__main__':
    main()