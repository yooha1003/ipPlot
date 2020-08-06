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
    [ver0.10]       release of this script (2020.08.07)

++ Copyright at uschoi@nict.go.jp / qtwing@naver.com ++
''')

parser.add_argument('--version', action='version', version='Version 0.1')
parser.parse_args()

# assign input arguments
img_input = str(input('++ Enter Nifti file name:  '))
# extract header information of input
# load mri image
img1 = nib.load(img_input)
# header information extraction
hdr = img1.header
x_pixdim,y_pixdim,z_pixdim = hdr.get_zooms()
min_pixdim = min(hdr.get_zooms())
max_pixdim = max(hdr.get_zooms())
ratio_pixdim = int(max_pixdim / min_pixdim)
ratio_pixdim_x_width = int(max_pixdim / x_pixdim)
ratio_pixdim_y_width = int(max_pixdim / y_pixdim)
ratio_pixdim_z_width = int(max_pixdim / z_pixdim)
ratio_pixdim_x_vline = int(x_pixdim / min_pixdim)
ratio_pixdim_y_vline = int(y_pixdim / min_pixdim)
ratio_pixdim_z_vline = int(z_pixdim / min_pixdim)

# dimension
img1_array = np.asarray(img1.dataobj)
max_val = np.ndarray.max(img1_array)
# x,y,z ranges
x_dim_original = img1_array.shape[0]
y_dim_original = img1_array.shape[1]
z_dim_original = img1_array.shape[2]

# other arguments
x_coord = int(input('+ Enter X coordinate (' + str(0) + '-' + str(x_dim_original) + '):  '))
y_coord = int(input('+ Enter Y coordinate (' + str(0) + '-' + str(y_dim_original) + '):  '))
z_coord = int(input('+ Enter Z coordinate (' + str(0) + '-' + str(z_dim_original) + '):  '))
vline1 = int(input('+ Selected horizontal line of 1st view (' + str(0) + '-' + str(z_dim_original) + '):  '))
vline2 = int(input('+ Selected horizontal line of 2nd view (' + str(0) + '-' + str(z_dim_original) + '):  '))
vline3 = int(input('+ Selected horizontal line of 3rd view (' + str(0) + '-' + str(y_dim_original) + '):  '))
cb_min = float(input('++ Minimum of an output scale (' + str(0) + '-' + str(max_val) + '):  '))
cb_max = float(input('++ Maximum of an output scale (' + str(0) + '-' + str(max_val) + '):  '))


# upscaling based on minimum pixel resolution
img1_up = resample_img(img1, target_affine=np.eye(3)*min_pixdim, interpolation='nearest')
img1_up_array = np.asarray(img1_up.dataobj)
# x,y,z ranges
x_dim = img1_up_array.shape[0]
y_dim = img1_up_array.shape[1]
z_dim = img1_up_array.shape[2]
max_dim = max(x_dim,y_dim,z_dim)
min_dim = min(x_dim,y_dim,z_dim)

## Main Run ##
vline1_up = round(vline1 * (z_dim / z_dim_original))
vline2_up = round(vline2 * (z_dim / z_dim_original))
vline3_up = round(vline3 * (y_dim / y_dim_original))

# data extraction for line plots
img1_up_array_x = img1_up_array[:,round(y_coord * (y_dim / y_dim_original)),vline1_up]
img1_up_array_y = img1_up_array[round(x_coord * (x_dim / x_dim_original)),:,vline2_up]
img1_up_array_z = img1_up_array[:,vline3_up,round(z_coord * (z_dim / z_dim_original))]

# subplot
fig = plt.figure(figsize=(12, 6))

# set th w_ratio
w_ratio = [x_dim / min_dim,y_dim / min_dim,x_dim / min_dim]

#
# grid spec
gs = gridspec.GridSpec(2, 3,
                       width_ratios=w_ratio,
                       height_ratios=[2,1]
                       )
gs.update(wspace=0.4,hspace=0.2)

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
ax3 = fig.add_subplot(gs[2])
ax4 = fig.add_subplot(gs[3])
ax5 = fig.add_subplot(gs[4])
ax6 = fig.add_subplot(gs[5])

# images
im1 = ax1.imshow(img1_up_array[:,round(y_coord * (y_dim / y_dim_original)),:].T,cmap='jet',origin='lower',vmin=cb_min, vmax=cb_max)
ax1.plot([0, x_dim], [vline1_up, vline1_up], color='#f9fc2b', linestyle='-')
ax1.set_facecolor('white')
ax1.grid(False)
ax1.autoscale(False)
plt.sca(ax1)
plt.xticks([0,round(x_dim/2),x_dim], [0,round(x_dim_original/2),round(x_dim_original)], color="black")
plt.yticks([0,round(z_dim/2),z_dim], [0,round(z_dim_original/2),round(z_dim_original)], color="black")

im2 = ax2.imshow(img1_up_array[round(x_coord * (x_dim / x_dim_original)),:,:].T,cmap='jet',origin='lower',vmin=cb_min, vmax=cb_max)
ax2.plot([0, y_dim], [vline2_up, vline2_up], color='#f9fc2b', linestyle='-')
ax2.set_facecolor('white')
ax2.grid(False)
ax2.autoscale(False)
plt.sca(ax2)
plt.xticks([0,round(y_dim/2),y_dim], [0,round(y_dim_original/2),round(y_dim_original)], color="black")
plt.yticks([0,round(z_dim/2),z_dim], [0,round(z_dim_original/2),round(z_dim_original)], color="black")

im3 = ax3.imshow(img1_up_array[:,:,round(z_coord * (z_dim / z_dim_original))].T,cmap='jet', origin='lower',vmin=cb_min, vmax=cb_max)
ax3.plot([0, x_dim], [vline3_up, vline3_up], color='#f9fc2b', linestyle='-')
ax3.set_facecolor('white')
ax3.grid(False)
ax3.autoscale(False)
plt.sca(ax3)
plt.xticks([0,round(x_dim/2),x_dim], [0,round(x_dim_original/2),round(x_dim_original)], color="black")
plt.yticks([0,round(y_dim/2),y_dim], [0,round(y_dim_original/2),round(y_dim_original)], color="black")

# subplots
ax4.plot(img1_up_array_x,color='#f9fc2b')
plt.sca(ax4)
plt.xticks([0,round(x_dim/2),x_dim], [0,round(x_dim_original/2),round(x_dim_original)], color="black")
# plt.minorticks_on()
plt.margins(x=0)
ax4.set_facecolor('gray')
ax4.grid(False)

ax5.plot(img1_up_array_y,color='#f9fc2b')
plt.sca(ax5)
plt.xticks([0,round(y_dim/2),y_dim], [0,round(y_dim_original/2),round(y_dim_original)], color="black")
# plt.minorticks_on()
plt.margins(x=0)
ax5.set_facecolor('gray')
ax5.grid(False)


ax6.plot(img1_up_array_z,color='#f9fc2b')
plt.sca(ax6)
plt.xticks([0,round(x_dim/2),x_dim], [0,round(x_dim_original/2),round(x_dim_original)], color="black")
# plt.minorticks_on()
plt.margins(x=0)
ax6.set_facecolor('gray')
ax6.grid(False)

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.83, 0.13, 0.03, 0.65])
cbar = fig.colorbar(im1, cax=cbar_ax)
# cbar = fig.colorbar(im1, cax=cbar_ax, ticks=[0,0.5,1,1.5,2.0])
cbar.set_label('A Range of Intensity Values', labelpad=25, rotation=270, size=15)
# save figure
plt.savefig('ipPlot_figure.png', dpi = 300)
