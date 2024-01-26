import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from scipy.ndimage import convolve


class HazeRemoval:
    def __init__(self, filename, w=0.8, r=60):
        self.filename = filename
        self.w = w
        self.r = r
        self.eps = 10 ** (-3)
        self.t = 0.10

    # 将一维索引转换为多维索引
    def index_1to2(self, array_shape, ind):
        rows = (ind.astype('int') / array_shape[1])
        cols = (ind.astype('int') % array_shape[1])
        return rows, cols

    # 将RGB图像转换为灰度图像
    def rgb_to_gray(self, rgb):
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

    # 去雾函数
    def haze_removal(self, save_path=None):
        ori_image = np.array(Image.open(self.filename))
        img = np.array(ori_image).astype(np.double) / 255.0
        gray_image = self.rgb_to_gray(img)
        # 暗通道
        dark_image = img.min(axis=2)
        # 大气光
        (i, j) = map(int, self.index_1to2(dark_image.shape, dark_image.argmax()))
        A = img[i, j, :].mean()
        # 透射率
        T = 1 - self.w * dark_image / A
        # 使用导向滤波对透射率进行平滑
        transmissionFilter = self.guided_filter(gray_image, T, self.r, self.eps)
        transmissionFilter[transmissionFilter < self.t] = self.t

        resultImage = np.zeros_like(img)
        for i in range(3):
            resultImage[:, :, i] = (img[:, :, i] - A) / transmissionFilter + A

        resultImage[resultImage < 0] = 0
        resultImage[resultImage > 1] = 1

        result_image = Image.fromarray((resultImage * 255).astype(np.uint8))
        # 保存图片
        if save_path:
            result_image.save(save_path)
        self.show_img(ori_image, result_image)

    # 显示图片
    def show_img(self, oriImage, result):
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.imshow(oriImage)
        plt.title('Original Image')

        plt.subplot(1, 2, 2)
        plt.imshow(result)
        plt.title('Defogging Image')

        plt.show()

    # 滤波函数
    def guided_filter(self, I, p, r, eps=1e-8):
        # 计算均值
        mean_I = convolve(I, np.ones((r, r)) / r ** 2, mode='reflect', cval=0)
        mean_p = convolve(p, np.ones((r, r)) / r ** 2, mode='reflect', cval=0)
        mean_Ip = convolve(I * p, np.ones((r, r)) / r ** 2, mode='reflect', cval=0)
        mean_II = convolve(I * I, np.ones((r, r)) / r ** 2, mode='reflect', cval=0)

        # 计算相关性
        cov_Ip = mean_Ip - mean_I * mean_p
        cov_II = mean_II - mean_I * mean_I

        # 计算权重和对目标图像进行滤波
        a = cov_Ip / (cov_II + eps)
        b = mean_p - a * mean_I
        mean_a = convolve(a, np.ones((r, r)) / r ** 2, mode='reflect', cval=0)
        mean_b = convolve(b, np.ones((r, r)) / r ** 2, mode='reflect', cval=0)
        q = mean_a * I + mean_b

        return q


# 测试函数
if __name__ == '__main__':
    imageName = "11.png"
    hz = HazeRemoval("Images/Hazeimg/" + imageName)
    save_path = "Images/Dehazeimg/new_" + imageName
    hz.haze_removal(save_path)
