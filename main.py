import cv2
import numpy as np

# ******* define a function that generates the noises ******* #
def noisy(noise_typ,image):

    # gaussian
    if noise_typ == "gauss":
        row,col= image.shape
        mean = 0
        var = 100
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col))
        gauss = gauss.reshape(row,col)

        noisy = (image + gauss)
        for i in range(len(noisy)):
            for j in range(len(noisy[i])):
                noisy[i,j] = np.uint8(noisy[i,j])
                if( noisy[i,j]<0):
                    noisy[i, j] = 0
                elif  noisy[i,j]>255:
                    noisy[i, j]=255
        cvuint8 = cv2.convertScaleAbs(noisy)
        return cvuint8
    # sault & pepper
    elif noise_typ == "s&p":
        s_vs_p = 0.5
        amount = 0.04
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        out[coords] = 255
        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))for i in image.shape]
        out[coords] = 0
        return out
    # poison
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals1 = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals1) / float(vals1)
        cvuint8 = cv2.convertScaleAbs(noisy)
        return cvuint8
    # speckle
    elif noise_typ =="speckle":
        row,col = image.shape
        gauss = np.random.randn(row,col)
        gauss = gauss.reshape(row,col)
        noisy = image + image * gauss
        cvuint8 = cv2.convertScaleAbs(noisy)
        return cvuint8

# ******* making noisy images ******* #

#Lenna image
img1 = cv2.imread('Lenna.png',0)
noisyImg1 = noisy('gauss',img1)

#cameraman image
img2 = cv2.imread('cameraman.png',0)
noisyImg2 = noisy('s&p',img2)

#lamborghini image
img3 = cv2.imread('lamborghini.png',0)
noisyImg3 = noisy('poisson',img3)

# ******* FILTERS ******* #

# median filter #
denoised_median = cv2.medianBlur(img1,3)

# mean filter #
kernel = np.ones((5,5),np.float32)/25
denoised_mean = cv2.filter2D(noisyImg1,-1,kernel)

# ******* Showing the result ******* #
cv2.imshow('noisy',noisyImg1)
cv2.imshow('denoised by median filter',denoised_median)
cv2.imshow('denoised by mean filter',denoised_mean)
cv2.waitKey()