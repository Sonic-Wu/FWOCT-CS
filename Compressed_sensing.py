#%%
import numpy as np
from pylbfgs import owlqn
from scipy.fftpack import dct, idct
import pywt
import cv2
import pickle
import math
import os
import matplotlib.pyplot as plt
#%%

def main():
    # define scanning constant
    SS_frequency = 100160.26 # 100k Hz A scan frequency
    delta_T      = 1 / SS_frequency # Δt


    # Lissajous parameter
    # x arm -> Asin(at + p)
    # y arm -> Bsin(bt)
    x_sin_frequency = 162.005 # 188Hz for x arm
    y_sin_freqeuncy = 183.999 # 190HZ for y arm
    B_scan_length   = 186712


    # mapping the Lissajous coordinate using A-scan interval
    T = np.arange(0, 1/SS_frequency * B_scan_length, 1/SS_frequency)


    # Wrapping Raw coordinate    
    X = np.around(np.sin(2 * math.pi * x_sin_frequency * T), decimals = 3)
    Y = np.around(np.sin(2 * math.pi * y_sin_freqeuncy * T), decimals = 3)
    #X = np.asarray(X, dtype = np.int16)
    #Y = np.asarray(Y, dtype = np.int16)
    coordinate = list(zip(X, Y))


    # Scale up the Lissajous coordinate into the integer
    size = 698
    X_scale = np.asarray((size/2) * X + (size/2), dtype = np.int16)
    Y_scale = np.asarray((size/2) * Y - (size/2), dtype = np.int16)
    grid = list(zip(X_scale, Y_scale))


    # Load in Lissajous sequence A-line
    os.chdir(r"/home/xwu25/Compressed sensing")
    print(os.getcwd())


    with open("Target10.pkl", 'rb') as file:
        B_scan_sequence_2D = pickle.load(file)

    row, column = B_scan_sequence_2D.shape
    for each_row in range(row):
        if not each_row == 613:
            continue
        B_Scan_sequence_1D = B_scan_sequence_2D[each_row]

        # Reconstuct sampled image by filing B_Scan pixel into grid
        image_nx = size + 2 # column
        image_ny = size + 2 # row
        sample_image = np.zeros((image_ny,image_nx), dtype = np.uint8)
        for i,each_pixel in enumerate(grid):
            sample_image[-each_pixel[1], each_pixel[0]] = B_Scan_sequence_1D[i]

        test = False 
        # compressed sensing using a Nikon microscopy target image
        if test:
            np.random.seed(1)
            original_image = cv2.imread('Target_10_Nikon_flawless.png', cv2.IMREAD_GRAYSCALE)
            image_ny, image_nx = original_image.shape
            sample_ratio = 0.6
            k = round(image_nx*image_ny*sample_ratio)
            ri = np.random.choice(image_nx*image_ny, k, replace = False)
            y = original_image.flatten("F")[ri]
            print("y length: {}".format(len(y)))
            y = np.expand_dims(y, axis = 1)
            sample_image = np.zeros(original_image.shape)
            sample_image.T.flat[ri] = y
            print("Sample Image Dimension: {}".format(sample_image.shape))
            print("OriginalImage Dimension:{}".format(original_image.shape))
            cv2.imwrite('Sampled_image_{}.png'.format(sample_ratio), sample_image)

            folder_name = r'/home/xwu25/Compressed sensing/Target_Nikon_flawless_test'
            if not os.path.exists(folder_name):
                os.mkdir(folder_name)
            os.chdir(folder_name)
            
        else:
            sample_image_flatten = sample_image.flatten('F')    # flatten the sample_image via append the column
            ri = sample_image_flatten.nonzero()[0]              # sample pixel index
            sample_ratio = len(ri)/len(sample_image_flatten)
            y = sample_image_flatten[ri]                        # b -> 1d array
            y = np.expand_dims(y, axis = 1)                     # b -> column vector
            #print("Observer Vector Y shape: {}".format(y.shape))
            folder_name = r'/home/xwu25/Compressed sensing/DWT'
            if not os.path.exists(folder_name):
                os.mkdir(folder_name)
            os.chdir(folder_name)

        
        # prepare wavelet transform (DWT) parameters
        global slices
        global wavelet_type
        wavelet_type = 'haar'
        dec_level = 2
        
        coeff_a = pywt.wavedec2(sample_image, wavelet = wavelet_type, level = dec_level)
        coeff_array, slices = pywt.coeffs_to_array(coeff_a)
        print("sample image slices: {}".format(slices))
        #cv2.imwrite("Coefficients_arr.png", coeff_array)

    
        coeff_arr_ny, coeff_arr_nx = coeff_array.shape
        print("Wavelet Coefficients array shape: {}".format(coeff_array.shape)) 
        
        # implement 2d DCT
        def dct2(a):
            return dct(dct(a.T, norm = 'ortho', axis = 0).T, norm = 'ortho', axis = 0)

        # implement 2d inverse DCT
        def idct2(a):
            return idct(idct(a.T, norm = 'ortho', axis = 0).T, norm = 'ortho', axis = 0) 

        # dct evaluate        
        def evaluate_dct(x, g, step):
            """An in-memory evaluation callback."""

            # we want to return two things: 
            # (1) the norm squared of the residuals, sum((Ax-b).^2), and
            # (2) the gradient 2*A'(Ax-b)

            # expand x columns-first
            x2 = x.reshape((image_nx, image_ny)).T

            # Ax is just the inverse 2D dct of x2
            Ax2 = idct2(x2)

            # stack columns and extract samples
            Ax = Ax2.T.flat[ri].reshape(y.shape)

            # calculate the residual Ax-b and its 2-norm squared
            Axb = Ax - y
            fx = np.sum(np.power(Axb, 2))

            # project residual vector (k x 1) onto blank image (ny x nx)
            Axb2 = np.zeros(x2.shape)
            Axb2.T.flat[ri] = Axb # fill columns-first

            # A'(Ax-b) is just the 2D dct of Axb2
            AtAxb2 = 2 * dct2(Axb2)
            AtAxb = AtAxb2.T.reshape(x.shape) # stack columns

            # copy over the gradient vector
            np.copyto(g, AtAxb)

            return fx

        # dwt evaluate
        def evaluate_dwt(x, g, step):
            """An in-memory evaluation callback."""
            # we want to return two things: 
            # (1) the norm squared of the residuals, sum((Ax-b).^2), and
            # (2) the gradient 2*A'(Ax-b)
            
            
            #print("Frequency domain Variable vector shape: {}".format(x.shape))
            # expand x columns-first
            x2 = x.reshape((coeff_arr_nx, coeff_arr_ny)).T
            x2_coeff = pywt.array_to_coeffs(x2, slices, output_format='wavedec2')
            #print("2D Frequency domain Variable shape: {}".format(x2.shape))

            
            # Ax is the inversed 2D wavelet transform of x2
            Ax2 = pywt.waverec2(x2_coeff, wavelet=wavelet_type)
            #print("Image shape after idwt: {}".format(Ax2.shape))
            
            
            # stack columns and extract samples
            Ax = Ax2.T.flat[ri].reshape(y.shape)

            
            # calculate the residual Ax-b and its 2-norm squared
            Axb = Ax - y
            fx = np.sum(np.power(Axb, 2))

            
            # project residual vector (k x 1) onto blank image (image_ny x image_nx)
            Axb2 = np.zeros((image_ny, image_nx))
            Axb2.T.flat[ri] = Axb # fill columns-first
            #print("Residual matrix shape (Ax-b): {}".format(Axb2.shape))

            
            # A'(Ax-b) is just the 2D dwt of Axb2
            AtAxb2_coeff = pywt.wavedec2(Axb2, wavelet = wavelet_type, level = dec_level)
            AtAxb2, coeff_slices = pywt.coeffs_to_array(AtAxb2_coeff)
            AtAxb2 = 2 * AtAxb2
            #AtAxb2 = np.asarray(AtAxb2, dtype = np.float32)
            #print("AtAxb2 shape: {}".format(AtAxb2.shape))
 
            AtAxb = AtAxb2.T.reshape(x.shape) # stack columns

            # copy over the gradient vector
            np.copyto(g, AtAxb)

            return fx
        
        
        # select transformation alogrithm
        Trans_Algo ='DWT'

        if Trans_Algo == 'DCT':
            # comment following lines when using DWT
            # perform the L1 minimization in memory
            param_lambda = 40
            Xat2 = owlqn(image_nx * image_ny, evaluate_dct, None, param_lambda)
            Xat = Xat2.reshape(image_nx, image_ny).T
            Xa  = idct2(Xat)
            
            # reconstruction
            image_reconstructed = Xa.astype('uint8')
            plt.imsave('CS_reconstruct_{}_{}X{}_lambda{}_DCT_Sample_{}.png'.format(each_row,image_ny,image_nx, param_lambda, np.around(sample_ratio, decimals = 2)),np.flip(image_reconstructed, 0),cmap = 'gray')
        
        if Trans_Algo == 'DWT':
            # perform the L1 minimization in memory
            param_lambda = 40
            Xat2_coeff = owlqn(coeff_arr_nx*coeff_arr_ny, evaluate_dwt, None, param_lambda)
            Xat_coeff = Xat2_coeff.reshape(coeff_arr_nx, coeff_arr_ny).T
            #cv2.imwrite('Coefficient_Array_lambda_{}_wavelet_{}_declevel_{}_Sample_{}_decimal_3.png'.format(step_size, wavelet_type,dec_level, np.around(sample_ratio, decimals = 2)), Xat_coeff)
            Xa_coeff = pywt.array_to_coeffs(Xat_coeff, slices, output_format = 'wavedec2')
            Xa = pywt.waverec2(Xa_coeff, wavelet=wavelet_type)
        
            # reconstruction
            image_reconstructed = Xa.astype('uint8')
            plt.imsave('CS_reconstruct_{}_{}X{}_lambda_{}_wavelet_{}_declevel_{}_Sample_{}_decimal_3.png'.format(each_row,image_ny,image_nx, param_lambda, wavelet_type,dec_level, np.around(sample_ratio, decimals = 2)),np.flip(image_reconstructed,0),cmap = 'gray', vmin = 50, vmax = 200)

            #cv2.imwrite('CS_reconstruct_613_{}X{}.png'.format(size,size), np.flip(image_reconstructed, 0))
if __name__ == "__main__":
    main()
#%%
