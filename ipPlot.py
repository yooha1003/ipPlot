#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# set the module
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import nibabel as nib
import argparse
import scipy.interpolate as interpolate
from nilearn.image import resample_img

# HELP Section
parser = argparse.ArgumentParser(description='## Intensity profile of NifTI file plotting script ##', formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog='''\
version history:
    [ver0.10]       release of this script (2020.08.04)

++ Copyright at uschoi@nict.go.jp / qtwing@naver.com ++
''')

parser.add_argument('--version', action='version', version='Version 0.1')
parser.parse_args()

# assign input arguments
img_input = str(input('++ Enter Nifti file name:  '))
x_coord = int(input('+ Enter X coordinate ?:  '))
y_coord = int(input('+ Enter Y coordinate ?:  '))
z_coord = int(input('+ Enter Z coordinate ?:  '))
vline_x = int(input('+ Selected horizontal line of 1st view ?:  '))
vline_y = int(input('+ Selected horizontal line of 2nd view ?:  '))
vline_z = int(input('+ Selected horizontal line of 3rd view ?:  '))
cb_min = int(input('++ Minimum of an output scale ?:  '))
cb_max = int(input('++ Maximum of an output scale ?:  '))

## Main Run ##
# load mri image
img1 = nib.load(img_input)
# header information extraction
hdr = img1.header
x_pixdim,y_pixdim,z_pixdim = hdr.get_zooms()
min_pixdim = min(hdr.get_zooms())
max_pixdim = max(hdr.get_zooms())
ratio_pixdim = int(max_pixdim / min_pixdim)
vline_y = vline_y * ratio_pixdim
# upscaling based on minimum pixel resolution
img1_up = resample_img(img1, target_affine=np.eye(3)*min_pixdim, interpolation='nearest')
img1_up_array = np.asarray(img1_up.dataobj)
# x,y,z ranges
x_dim = img1_up_array.shape[0]
y_dim = img1_up_array.shape[1]
z_dim = img1_up_array.shape[2]
# # data extraction
img1_up_array_x = img1_up_array[x_coord,:,vline_x]
img1_up_array_y = img1_up_array[:,y_coord,vline_y]
img1_up_array_z = img1_up_array[:,vline_z,z_coord]

# subplot
fig = plt.figure(figsize=(12, 6))

if ratio_pixdim == 1:
    w_ratio = [2,2,2]
else:
    w_ratio = [1,2,2]

gs = gridspec.GridSpec(2, 3,
                       width_ratios=w_ratio,
                       height_ratios=[2,1]
                       )
gs.update(wspace=0.4, hspace=0.2)

ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1])
ax3 = plt.subplot(gs[2])
ax4 = plt.subplot(gs[3])
ax5 = plt.subplot(gs[4])
ax6 = plt.subplot(gs[5])

# images
im1 = ax1.imshow(img1_up_array[x_coord,:,:].T,cmap='jet',origin='lower',vmin=cb_min, vmax=cb_max)
ax1.plot([0, y_dim], [vline_x, vline_x], color='#f9fc2b', linestyle='-')
ax1.set_facecolor('white')
ax1.grid(False)
im2 = ax2.imshow(img1_up_array[:,y_coord,:].T,cmap='jet',origin='lower',vmin=cb_min, vmax=cb_max)
ax2.plot([0, x_dim], [vline_y, vline_y], color='#f9fc2b', linestyle='-')
ax2.set_facecolor('white')
ax2.grid(False)
im3 = ax3.imshow(img1_up_array[:,:,z_coord].T,cmap='jet', origin='lower',vmin=cb_min, vmax=cb_max)
ax3.plot([0, x_dim], [vline_z, vline_z], color='#f9fc2b', linestyle='-')
ax3.set_facecolor('white')
ax3.grid(False)
plt.sca(ax1)
plt.xticks([0,round(y_dim/2),y_dim], [0,round(y_dim/(2*ratio_pixdim)),round(y_dim/ratio_pixdim)], color="black")
plt.sca(ax3)
plt.yticks([0,round(y_dim/2),y_dim], [0,round(y_dim/(2*ratio_pixdim)),round(y_dim/ratio_pixdim)], color="black")

# plots

ax4.plot(img1_up_array_x,color='#f9fc2b')
plt.sca(ax4)
plt.xticks([0,round(y_dim/2),y_dim], [0,round(y_dim/4),round(y_dim/2)], color="black")
ax4.set_facecolor('gray')
ax4.grid(False)
ax5.plot(img1_up_array_y,color='#f9fc2b')
ax5.set_facecolor('gray')
ax5.grid(False)
ax6.plot(img1_up_array_z,color='#f9fc2b')
ax6.set_facecolor('gray')
ax6.grid(False)

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.83, 0.13, 0.03, 0.72])
cbar = fig.colorbar(im1, cax=cbar_ax)
# cbar = fig.colorbar(im1, cax=cbar_ax, ticks=[0,0.5,1,1.5,2.0])
cbar.set_label('A Range of Intensity Values', labelpad=25, rotation=270, size=15)
# save figure
plt.savefig('ipPlot_figure.png', dpi = 300)
