#DARN_pulses
#Created: January, 2021 by Gareth Perry
#
#Purpose of script is to package up SuperDARN pulses for a given RRI lv1 file and return voltage, ephemeris, and other pertinent information so that the pulses can be compared to ray tracing estimates.


#import pdb; pdb.set_trace()

import os
os.chdir('/Users/perry/GitHub/ePOP-RRI/')

import numpy as np 
import h5py
import datetime, time 

from scipy import signal,interpolate


import matplotlib.pyplot as plt

from RRI import *


#fn1_=RRI('/Users/perry/Downloads/RRI_20150813_192914_193311_lv1_v5.h5') #the RRI file that we're interested in

#fn1_=RRI('/Users/perry/Downloads/RRI_20150618_032244_032641_lv1_v5.h5')

#fn1_=RRI('/Users/perry/Downloads/RRI_20150507_084944_085541_lv1_v5.h5')
fn1_=RRI('/Users/perry/Downloads/RRI_20170805_183314_183711_lv1_v5.h5')


#import pdb; pdb.set_trace()

#fn_out="/Users/perry/Google Drive File Stream/My Drive/RayTraceModel/Ruzic_rays/dat/DARN_pulses_out_14042017.h5"
fn_out="/Users/perry/Downloads/DARN_pulses_out_05082017.h5"


pls_=DARN_pulse_seeker(fn1_.m1_mV+1j*fn1_.m2_mV) #use the pulse seeker to extract the indices of the pulses along with other useful information

#interpolations here based on RRI sample number
glat_interp=interpolate.interp1d(np.arange(len(fn1_.glat_))*fn1_.fs_,fn1_.glat_,fill_value='extrapolate') #1D interpolate for sample number and spacecraft latitude
glon_interp=interpolate.interp1d(np.arange(len(fn1_.glon_))*fn1_.fs_,fn1_.glon_,fill_value='extrapolate') #1D interpolate for sample number and spacecraft longitude
alt_interp=interpolate.interp1d(np.arange(len(fn1_.alt_))*fn1_.fs_,fn1_.alt_,fill_value='extrapolate') #1D interpolate for sample number and spacecraft altitude

pitch_interp=interpolate.interp1d(np.arange(len(fn1_.pitch_))*fn1_.fs_,fn1_.pitch_,fill_value='extrapolate') #1D interpolate for sample number and spacecraft pitch
yaw_interp=interpolate.interp1d(np.arange(len(fn1_.yaw_))*fn1_.fs_,fn1_.yaw_,fill_value='extrapolate') #1D interpolate for sample number and spacecraft yaw
roll_interp=interpolate.interp1d(np.arange(len(fn1_.roll_))*fn1_.fs_,fn1_.roll_,fill_value='extrapolate') #1D interpolate for sample number and spacecraft roll

met_interp=interpolate.interp1d(np.arange(len(fn1_.epop_met))*fn1_.fs_,fn1_.epop_met,fill_value='extrapolate') #1D interpolate for sample number and mission elapsed time


#MATLAB plotting comparison program is expecting 90 x n arrays of parameters, where 'n' is the number of pulses.
glat_out=np.zeros((90,len(pls_[0])))
glon_out=np.zeros((90,len(pls_[0])))
alt_out=np.zeros((90,len(pls_[0])))

pitch_out=np.zeros((90,len(pls_[0])))
yaw_out=np.zeros((90,len(pls_[0])))
roll_out=np.zeros((90,len(pls_[0])))

met_out=np.zeros((90,len(pls_[0])))

d1_out=np.zeros((90,len(pls_[0]))) #dipole 1 voltage (mV)
d2_out=np.zeros((90,len(pls_[0]))) #dipole 2 voltage (mV)

m1_out=np.zeros((90,len(pls_[0]))) #monopole 1 voltage (mV)
m2_out=np.zeros((90,len(pls_[0]))) #monopole 2 voltage (mV)
m3_out=np.zeros((90,len(pls_[0]))) #monopole 3 voltage (mV)
m4_out=np.zeros((90,len(pls_[0]))) #monopole 4 voltage (mV)

#loop through all of the detected pulses and select the 90 surrounding samples
for i in range(len(pls_[0])):

	smps_=np.arange(90)+pls_[0][i]-45 #the 90 surrounding samples

	glat_out[:,i]=glat_interp(smps_)
	glon_out[:,i]=glon_interp(smps_)
	alt_out[:,i]=alt_interp(smps_)
	
	pitch_out[:,i]=pitch_interp(smps_)
	yaw_out[:,i]=yaw_interp(smps_)
	roll_out[:,i]=roll_interp(smps_)

	met_out[:,i]=met_interp(smps_)

	d1_out[:,i]=np.abs(fn1_.m1_mV[smps_]+1j*fn1_.m2_mV[smps_])
	d2_out[:,i]=np.abs(fn1_.m3_mV[smps_]+1j*fn1_.m4_mV[smps_])

	m1_out[:,i]=fn1_.m1_mV[smps_]
	m2_out[:,i]=fn1_.m2_mV[smps_]
	m3_out[:,i]=fn1_.m3_mV[smps_]
	m4_out[:,i]=fn1_.m4_mV[smps_]

#import pdb; pdb.set_trace()

#export the information in an h5 file
hf=h5py.File(fn_out,'w')

hf.create_dataset('pulse_ids',data=pls_[0])

hf.create_dataset('pulse_glat',data=glat_out)
hf.create_dataset('pulse_glon',data=glon_out)
hf.create_dataset('pulse_alt',data=alt_out)

hf.create_dataset('pulse_pitch',data=pitch_out)
hf.create_dataset('pulse_yaw',data=yaw_out)
hf.create_dataset('pulse_roll',data=roll_out)

hf.create_dataset('pulse_met',data=met_out)

hf.create_dataset('pulse_d1_mvolts',data=d1_out)
hf.create_dataset('pulse_d2_mvolts',data=d2_out)

hf.create_dataset('pulse_m1_mvolts',data=m1_out)
hf.create_dataset('pulse_m2_mvolts',data=m2_out)
hf.create_dataset('pulse_m3_mvolts',data=m3_out)
hf.create_dataset('pulse_m4_mvolts',data=m4_out)

hf.close()










