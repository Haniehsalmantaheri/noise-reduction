import cv2
import numpy as np
from matplotlib import pyplot as plt

# ******* define a function that generates the noises ******* #
def noisy(noise_typ,image_in):

    # gaussian
    if noise_typ == "gauss":
        row,col= image_in.shape
        mean = 0
        var = 100
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col))
        gauss = gauss.reshape(row,col)

        noisy = (image_in + gauss)
        for i in range(len(noisy)):
            for j in range(len(noisy[i])):
                noisy[i,j] = np.uint8(noisy[i,j])
                if( noisy[i,j]<0):
                    noisy[i,j] = 0
                elif  noisy[i,j]>255:
                    noisy[i,j]=255
        cvuint8 = cv2.convertScaleAbs(noisy)
        return cvuint8
    # sault & pepper
    elif noise_typ == "s&p":
        s_vs_p = 0.5
        amount = 0.04
        out = np.copy(image_in)
        # Salt mode
        num_salt = np.ceil(amount * image_in.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image_in.shape]
        out[coords] = 255
        # Pepper mode
        num_pepper = np.ceil(amount* image_in.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))for i in image_in.shape]
        out[coords] = 0
        return out
    # speckle
    elif noise_typ =="speckle":
        row,col = image_in.shape
        gauss = np.random.randn(row,col)
        gauss = gauss.reshape(row,col)
        noisy = image_in + image_in * gauss
        cvuint8 = cv2.convertScaleAbs(noisy)
        return cvuint8

# ******* making noisy images ******* #

#Lenna image
img1 = cv2.imread('Lenna.png',0)
noisyImg1 = noisy('gauss',img1)
#dst = cv2.fastNlMeansDenoisingColored(noisyImg1,None,10,10,7.21)

#cameraman image
img2 = cv2.imread('cameraman.png',0)
noisyImg2 = noisy('s&p',img2)

#lamborghini image
img3 = cv2.imread('lamborghini.png',0)
noisyImg3 = noisy('speckle',img3)

#images with periodic noise

img = cv2.imread('test1.jpg',0)
img4 = cv2.imread('test2.jpg',0)

# ************** FILTERS ************** #

###**Spetial Filters**###

# median filter #
denoised_median = cv2.medianBlur(noisyImg1,3)

# mean filter #
kernel = np.ones((6,6),np.float32)/36
denoised_mean = cv2.filter2D(noisyImg1,-1,kernel)

###**Frequency Domain**####

dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

rows, cols = img.shape
crow,ccol = rows/2 , cols/2
crow = int(crow)
ccol = int(ccol)

# create a mask first, center square is 1, remaining all zeros
mask = np.zeros((rows,cols,2),np.uint8)
mask[crow-30:crow+30, ccol-30:ccol+30] = 1

# apply mask and inverse DFT
fshift = dft_shift*mask
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

# ******* Showing the result ******* #

cv2.imshow('noisy',noisyImg1)
cv2.imshow('denoised by median filter',denoised_median)
cv2.imshow('denoised by mean filter',denoised_mean)
cv2.waitKey()

plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img_back, cmap = 'gray')
plt.title('Output Image'), plt.xticks([]), plt.yticks([])
plt.show()