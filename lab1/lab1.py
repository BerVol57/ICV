import numpy as np
import cv2 as cv
#photos
img_friend = cv.imread('input/friend.jpg')
img_bananas = cv.imread('input/bananas.jpg')
#text
amazing_text = cv.imread('input/amazing_text.jpg')
text = cv.imread('input/text.jpg')
#Перетворення у чб зображення
def _2grayscale(img=np.ndarray):
    I=np.dot(img[..., :3],
             [0.36, 0.53, 0.11])
    return I.astype(np.uint8)
#Гістограма
def histogram(img=np.ndarray):
    I = _2grayscale(img)
    H=[]
    for i in range(256):
        H=np.append(H, np.count_nonzero(I==i))
    return H
#Ймовірність
def p(img=np.ndarray):
    H_sum = histogram(img).sum()
    H = histogram(img)
    return H/H_sum
#сумарні вірогідності двох груп
def q1(img=np.ndarray, t=int):
    if p(img)[:t].sum()!=0:
        return p(img)[:t].sum()
    else: return 10
def q2(img=np.ndarray, t=int):
    if p(img)[ t+1: int(np.max(_2grayscale(img)))].sum()!=0:
        return p(img)[ t+1: int(np.max(_2grayscale(img)))].sum()
    else: return 10
#середні значення груп
def m1(img=np.ndarray, t=int):
    return np.dot(np.arange(256)[:t], p(img)[:t])/q1(img, t)
def m2(img=np.ndarray, t=int):
    return (np.dot(np.arange(256)[t+1: int(np.amax(_2grayscale(img)))],
                  p(img)[ t+1: int(np.max(_2grayscale(img)))]) ) /q2(img, t)
#дисперсія
def dysp(img, t):
    return q1(img, t)*q2(img, t)*((m1(img, t)-m2(img, t))**2)
#границя
def Topt(img=np.ndarray):
    t_arr = np.fromiter((dysp(img, x) for x in range(256)), np.dtype(np.int8))
    print(t_arr.argmin())
    return np.argmin(t_arr)
#маска
def create_mask(img=np.ndarray, cc=0):
    t_opt = Topt(img)
    I=_2grayscale(img)
    coord = np.where(I > t_opt)
    img[coord]=[cc, cc, cc]
    return img
#
cv.imshow('grayscale', _2grayscale(text))
cv.imshow('mask', create_mask(text))
