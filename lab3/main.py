import cv2 as cv
import numpy as np

pixelart = cv.imread('D:/Prog/ICV/lab3/input/pixelart.jpg')

kernel_matrix = np.ones((7, 7))


def resizing_img(img: np.ndarray, percent: float) -> np.ndarray:
    return cv.resize(img, (np.array(img.shape[:2]) * percent)[::-1].astype(int))


def pading_img(img: np.ndarray, width: float) -> np.ndarray:
    return np.pad(img, ((width, width),
                        (width, width),
                        (0, 0)),
                  mode='edge')


def morphology_operation(img: np.ndarray, kernel: np.ndarray, mode='e') -> np.ndarray:
    kw = kernel.shape[0]  # kernel width
    img = (img + 1) * (-1) % 256  # inverse img
    padded_img = pading_img(img, int(kw / 2))
    result_img = np.empty(img.shape)
    # check mode
    if mode == 'e':
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                if np.vdot(padded_img[x:x + kw, y:y + kw, 0], kernel) == kernel.sum() * 255:
                    result_img[x, y] = [255, 255, 255]
                else:
                    result_img[x, y] = [0, 0, 0]
    elif mode == 'd':
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                if np.vdot(padded_img[x:x + kw, y:y + kw, 0], kernel) > 0:
                    result_img[x, y] = [255, 255, 255]
                else:
                    result_img[x, y] = [0, 0, 0]
    else:
        print("Error: mode")
    return ((result_img + 1) * (-1) % 256).astype('uint8')  # inverse back


def closing(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return morphology_operation(morphology_operation(img, kernel, 'd'), kernel, 'e')


def opening(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return morphology_operation(morphology_operation(img, kernel, 'e'), kernel, 'd')


resized_img = resizing_img(pixelart, 0.3)
save_directory = 'D:/Prog/ICV/lab3/output/'
# cv.imwrite(save_directory+'img_org', pixelart)
cv.imwrite(save_directory+'img_e.jpg', morphology_operation(pixelart, kernel_matrix, 'e'))
cv.imwrite(save_directory+'img_d.jpg', morphology_operation(pixelart, kernel_matrix, 'd'))
cv.imwrite(save_directory+'img_closing.jpg', closing(pixelart, kernel_matrix))
cv.imwrite(save_directory+'img_opening.jpg', opening(pixelart, kernel_matrix))

cv.waitKey(0)
cv.destroyAllWindows()
