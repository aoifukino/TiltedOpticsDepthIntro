import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import disparity
from scipy import optimize
import pandas as pd

def get_disparity_plot(disp,low=0.4,height=5):
    # 視差の閾値

    # lowより大きいor小さい視差を取得
    masked_array = np.where(disp > low, disp, np.nan)
    masked_array_neg =  np.where(disp < -low, disp, np.nan)

    # x軸方向に平均を算出
    disp_plot = np.nanmean(masked_array,axis=1)
    disp_plot_neg = np.nanmean(masked_array_neg, axis=1)

    # 1次元配列に変換
    disp_plot = disp_plot.flatten()
    disp_plot_neg = disp_plot_neg.flatten()

    # 視差がheightより大きいor小さいものをnanに変換
    masked_disp_plot = np.where((disp_plot < height) & (disp_plot > -height), disp_plot, np.nan)
    masked_disp_plot_neg = np.where((disp_plot_neg < height) & (disp_plot_neg > -height), disp_plot_neg, np.nan)

    return masked_disp_plot, masked_disp_plot_neg

# 直線のモデル関数
def linear_model(params, x):
    m, b = params
    return m * x + b

# 残差関数
def residuals(params, x,y):
    return y - linear_model(params, x)

def main():
    print("===== START =====")
    # load image from images folder
    image_file_paths = [f for f in os.listdir('images') if f.endswith(('.jpg', '.jpeg'))]
    
    # パラメータを格納するDataFrame(左から、ゼロ視差、ゼロ視差(負の値)、直線近似の傾き、直線近似の傾き(負の値))
    df_params = pd.DataFrame(columns=['depth','zero_disp','zero_disp_neg','a','a_neg'])

    disp_list = [0,0.5,1,1.5,2]
    columns = ['depth']
    #concat disp_list and column
    for disp in disp_list:
        columns.append(f'{disp}')
    # 各視差の値も保持（距離推定式の確認用）
    df_depthAndDisp = pd.DataFrame(columns=columns)


    for path in image_file_paths:
        print("===== " + path + " =====")
        image = cv2.imread('images/' + path)
        image = cv2.resize(image, (352, 528))

        # 距離情報を取得
        filename = os.path.splitext(path)[0]


        # 赤、青のチャンネルに分割
        blue, _, red = cv2.split(image)

        # ===視差を算出する===
        disp = disparity.calcDisparityByZNCC(red, blue, 17, 21)

        # ===視差のプロットを取得===
        disp_plot,disp_plot_neg = get_disparity_plot(disp,low=0.3)

        # ===nanを除去した視差のみの配列と、そのインデックスの配列を取得===

        # nanでない領域のマスクを作成
        valid_indices = ~np.isnan(disp_plot)
        valid_indices_neg = ~np.isnan(disp_plot_neg)

        # nanでない領域のインデックスを取得
        y_disp_plot = np.flatnonzero(np.nan_to_num(disp_plot))
        y_disp_plot_neg = np.flatnonzero(np.nan_to_num(disp_plot_neg))

        # nanでない領域の視差のみの配列を取得
        x_disp_plot = disp_plot[valid_indices]
        x_disp_plot_neg = disp_plot_neg[valid_indices_neg]

        # ===最小二乗法による直線近似===
        #初期値
        param = [1,1]
        result = optimize.least_squares(residuals, param, args=(x_disp_plot,y_disp_plot),method='trf',loss='soft_l1')
        result_neg = optimize.least_squares(residuals, param, args=(x_disp_plot_neg,y_disp_plot_neg),method='trf',loss='soft_l1')

        a,b = result.x
        a_neg,b_neg = result_neg.x
        print(f"傾き:{a},切片:{b}")
        print(f"傾き:{a_neg},切片:{b_neg}")

        # ===視差が0の座標を取得===
        zero_disp = b
        zero_disp_neg = b_neg

        print(f"視差が0の座標{zero_disp}")
        print(f"視差が0の座標{zero_disp_neg}")

        disp_y_list = [ (d-b_neg)/a_neg for d in disp_list]
        disp_y_list.insert(0,filename)

        df_params = pd.concat([df_params,pd.DataFrame([[filename,zero_disp,zero_disp_neg,a,a_neg]],columns=df_params.columns)],ignore_index=True)
        df_depthAndDisp = pd.concat([df_depthAndDisp,pd.DataFrame([disp_y_list],columns=df_depthAndDisp.columns)],ignore_index=True)
        print("***df_params***")
        print(df_params)
        print("***df_depthAndDisp***")
        print(df_depthAndDisp)

        # ===グラフ描画===
        height = image.shape[0]
        
        plt.plot(-(b-range(height))/a,range(height),label='fitting')
        plt.plot(-(b_neg-range(height))/a_neg,range(height),label='fitting')

        plt.scatter(disp_plot,range(height))
        plt.scatter(disp_plot_neg,range(height))
        plt.xlim(-6,6)
        plt.ylim(0,height)
        plt.show()
    print("===== END =====")
    df_params.to_csv('csv/params.csv')
    # df_depthAndDisp.to_csv('csv/depthAndDisp2.csv')


if __name__ == "__main__":
    main()