import numpy as np
import cv2 as cv
# Матриці

move_20down_and_10right = np.zeros((41, 41))
move_20down_and_10right[0, 9] = 1

invers_matrix = np.zeros((3, 3))
invers_matrix[1, 1] = -1

diag_moving_blur = np.eye(7)

increasing_sharpness = np.array([[0, -1, 0],
                                 [-1, 5, -1],
                                 [0, -1, 0]])

sobelya = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]])

edge = np.array([[-1, -1, -1],
                 [-1, 8, -1],
                 [-1, -1, -1]])

amazing_matrix = np.array([[0.65418401, 0.79062252, 0.81479164],
                           [0.41451346, 0.01567841, 0.35951868],
                           [0.52360809, 0.10842763, 0.97811124]])


def gaus_kernel_func(x: int, y: int) -> np.float64:
    return (1 / (np.pi * 2)) * np.exp(-1 * (abs(x - y) ** 2) / 2)


gaus_smoothing_kernel = np.empty((11, 11))
for x__ in range(11):
    for y__ in range(11):
        gaus_smoothing_kernel[x__, y__] = gaus_kernel_func(x__, y__)

# Завантаження вхідного зображення
pixelart = cv.imread('D:/Prog/ICV/lab2/input/pixelart.jpg')

resized_tuple = tuple((np.array(pixelart.shape[:2]) * 0.3).astype(int))[::-1]
resized_img = cv.resize(pixelart, resized_tuple)


def pading_img(img: np.ndarray, width: float) -> np.ndarray:
    return np.pad(img, ((width, width),
                        (width, width), (0, 0)))


def spec_dot(img: np.ndarray, kernel: np.ndarray, d=1) -> np.ndarray:
    kw = kernel.shape[0]  # kernel width
    pad_img = pading_img(img, int(kw / 2))
    result_img = np.ndarray(img.shape)
    # Проходимося по пікселям зображення
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            for k in range(3):
                result_img[x, y][k] = int(np.vdot(pad_img[x:x + kw, y:y + kw, k], kernel) / d)
    return (result_img % 256).astype('uint8')


save_directory = 'D:/Prog/ICV/lab2/output/'

# cv.imwrite(save_directory + 'move_20down_and_10right.jpg', spec_dot(pixelart, move_20down_and_10right))
# cv.imwrite(save_directory + 'invers_matrix.jpg', spec_dot(pixelart, invers_matrix))
# cv.imwrite(save_directory + 'gaus_smoothing_kernel.jpg',
#            spec_dot(pixelart, gaus_smoothing_kernel, gaus_smoothing_kernel.sum()))
# cv.imwrite(save_directory + 'diag_moving_blur.jpg', spec_dot(pixelart, diag_moving_blur, 7))
# cv.imwrite(save_directory + 'increasing_sharpness.jpg',
#            spec_dot(spec_dot(pixelart, increasing_sharpness), np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]), 16))
# cv.imwrite(save_directory + 'sobelya.jpg', spec_dot(pixelart, sobelya))
# cv.imwrite(save_directory + 'edge.jpg', spec_dot(pixelart, edge))
# cv.imwrite(save_directory + 'amazing_matrix.jpg', spec_dot(pixelart, amazing_matrix))
