import numpy as np
import matplotlib.pyplot as plt
from skimage import io, util, filters, metrics
import cv2
from prettytable import PrettyTable

width = 49
resolution = 6000
delta = round((width/ resolution), 4) #период дискретизации
freq_diskr = round(1 / delta, 1) #частота дискретизации
coef = 1
freq = freq_diskr
phase= 0

n_px = 15 #Число пикселей в линейке регистрирующей среды -> выходной размер изображения
x = np.arange(0, delta * n_px, delta / 100) # значения по x
y=0.5 + 0.5*np.sin(freq * x + phase) 
fig = plt.subplots()
plt.plot(x, y)
plt.show()
print(delta)
print(freq_diskr)

def function(x):
    return 0.5 + 0.5*np.sin(freq * x + phase)
    # return np.sin(2*np.pi*frequency*x+phase*np.pi)/2+0.5 
counts_num = 50 

# Функция дискретизации 
def sampling(s_frequancy):
    out = []
    x = 0
    while x < counts_num:
        y = function(x)
        y1 = function(x + s_frequancy)
        out.append(round((y + y1) / 2, 2))
        x += s_frequancy
    return np.array(out)

# Преобразование дискретных значений в изображение
def image(row): 
    row = np.around(255 * row)
    image = np.full((row.size, row.size), row)
    return image

img1 = image(sampling(0.1))
img2 = image(sampling(0.2))
img3 = image(sampling(0.4))
img4 = image(sampling(0.8))
img5 = image(sampling(1))
img6 = image(sampling(2))
img7 = image(sampling(4))

fig = plt.figure()

fig, ([ax1, ax2, ax3, ax4, ax5, ax6,ax7] ) = plt.subplots(
    nrows=1, ncols=7,
    figsize=(18, 14)
)

ax1.set_title('01')
ax2.set_title('02')
ax3.set_title('03')
ax4.set_title('04')
ax5.set_title('05')
ax6.set_title('06')
ax7.set_title('07')


ax1.imshow(img1, cmap='gray')
ax2.imshow(img2, cmap='gray')
ax3.imshow(img3, cmap='gray')
ax4.imshow(img4,cmap='gray')
ax5.imshow(img5, cmap='gray')
ax6.imshow(img6, cmap='gray')
ax7.imshow(img7,cmap='gray')

plt.show()

# Изменение размера - интерполяция методом ближайшего соседа
def interpolation(img, size):
    out = cv2.resize(img, size, cv2.INTER_NEAREST)
    return out

img1Intr = interpolation(img1, img1.shape)
img2Intr = interpolation(img2, img1.shape)
img3Intr = interpolation(img3, img1.shape)
img4Intr = interpolation(img4, img1.shape)
img5Intr = interpolation(img5, img1.shape)
img6Intr = interpolation(img6, img1.shape)
img7Intr = interpolation(img7, img1.shape)

fig = plt.figure()

fig, ([ax1, ax2, ax3, ax4, ax5, ax6,ax7] ) = plt.subplots(
    nrows=1, ncols=7,
    figsize=(18, 14)
)

ax1.set_title('01')
ax2.set_title('02')
ax3.set_title('03')
ax4.set_title('04')
ax5.set_title('05')
ax6.set_title('06')
ax7.set_title('07')


ax1.imshow(img1Intr, cmap='gray')
ax2.imshow(img2Intr, cmap='gray')
ax3.imshow(img3Intr, cmap='gray')
ax4.imshow(img4Intr,cmap='gray')
ax5.imshow(img5Intr, cmap='gray')
ax6.imshow(img6Intr, cmap='gray')
ax7.imshow(img7Intr,cmap='gray')

plt.show()

images = [img1Intr, img2Intr, img3Intr, img4Intr, img5Intr, img6Intr, img7Intr]

for i in range(len(images)): 
    mse = round(metrics.mean_squared_error(img1, images[i]) / 255, 3) 
    print(mse)

for i in range(len(images)):  
    psnr = round(metrics.peak_signal_noise_ratio(img1, images[i], data_range=255), 3)
    print(psnr)

for i in range(len(images)):
    ssim = round(metrics.structural_similarity(img1, images[i], multichannel=True), 3)
    print(ssim)