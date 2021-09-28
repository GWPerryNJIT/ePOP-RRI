#class created for e-POP RRI instrument 
#Created: May, 2020 by Gareth Perry

#import pdb; pdb.set_trace()

#===Example Invocation===

#from RRI import *
#fn1_=RRI('/path/to/data/RRI_20200526_000904_001902_lv1_v5.h5') #load up the RRI data
#glat_=fn1_.glat_ for the spacecraft geographic latitude information, or
#m1_=fn1_.m1_mV for the monopole 1 measurements (in mV)


import numpy as np 
import h5py
import datetime, time 

from scipy import signal

fs=62500.33933 #RRI's sampling frequency


def Rotx(angle_): #function returning X-rotation matrix for a given angle
	Rx_=np.array([[1,0,0],[0,np.cos(np.deg2rad(angle_)),-np.sin(np.deg2rad(angle_))],[0,np.sin(np.deg2rad(angle_)), np.cos(np.deg2rad(angle_))]])#	
	return Rx_

def Roty(angle_): #function returning Y-rotation matrix for a given angle
	Ry_=np.array([[np.cos(np.deg2rad(angle_)),0,np.sin(np.deg2rad(angle_))],[0,1,0],[-np.sin(np.deg2rad(angle_)),0,np.cos(np.deg2rad(angle_))]])#	
	return Ry_

def Rotz(angle_): #function returning Z-rotation matrix for a given angle
	Rz_=np.array([[np.cos(np.deg2rad(angle_)),-np.sin(np.deg2rad(angle_)),0],[np.sin(np.deg2rad(angle_)),np.cos(np.deg2rad(angle_)),0],[0,0,1]])#	
	return Rz_


def DARN_pulse_seeker(rri_dat):

	#rri_dat is assumed to be complex
	#switch NaNs to 0
	loc_nan=np.isnan(rri_dat)
	rri_dat[loc_nan]=0


	#use the find peaks functionality whose indices correspond to the maximum.

	#distance == 1500 us lag assumption, equivalent to 94 samples at 62500.33933 Hz sampling
	#width == 300 us, a mimimum pulse length, equivalent to 18.75 samples at 62500.33933 Hz sampling
	#height == "Required height of peaks." is calculated by considering the height of the peak above the background noise, takeng o
	
	#find_peaks returns a dictionary with the following properties
	"""
	‘peak_heights’
	If height is given, the height of each peak in x.
	
	‘left_thresholds’, ‘right_thresholds’
	If threshold is given, these keys contain a peaks vertical distance to its neighbouring samples.
	
	‘prominences’, ‘right_bases’, ‘left_bases’
	If prominence is given, these keys are accessible. See peak_prominences for a description of their content.
	
	‘width_heights’, ‘left_ips’, ‘right_ips’
	If width is given, these keys are accessible. See peak_widths for a description of their content.
	
	‘plateau_sizes’, left_edges’, ‘right_edges’
	If plateau_size is given, these keys are accessible and contain the indices of a peak’s edges (edges are still part of the plateau) and the calculated plateau sizes.
	
	New in version 1.2.0.
	To calculate and return properties without excluding peaks, provide the open interval (None, None) as a value to the appropriate argument (excluding distance).
	"""

	pks_=signal.find_peaks(np.abs(rri_dat),height=np.nanquantile(np.abs(rri_dat),0.5),width=15,distance=94)

	return pks_ #return the indices of peak  associated with the pulses


class RRI:
	
	def __init__(self,filename):
		
		#This first bunch of code reads in the RRI data from the .h5 files and create several useful instances with that data.
		self.filename=filename

		self.fs_=fs #RRI sampling frequency

		#read in .h5 file
		self.data=h5py.File(filename,'r')

		#DATE IMFORMATION

		#MET epoch
		epop_epoch=datetime.datetime(1968,5,24,0,0,0)
		epop_epoch_s=time.mktime(epop_epoch.timetuple())#seconds since January 1, 1970 (Unix Epoch)

		
		epop_met=self.data.get("CASSIOPE Ephemeris/Ephemeris MET (seconds since May 24, 1968)")
		self.epop_met=np.array(epop_met).flatten()
		
		epop_s=np.array(self.epop_met)+epop_epoch_s-1 #epop seconds in Unix epoch

		#ePOP datettime object for this file in seconds resolution
		self.epop_dt=datetime.datetime.fromtimestamp(self.epop_met[0])+np.arange(len(self.epop_met))*datetime.timedelta(seconds=1) 
		
		
		#EPHEMERIS INFORMATION
		alt_=self.data.get("CASSIOPE Ephemeris/Altitude (km)")
		self.alt_=np.array(alt_).flatten()

		gei_=self.data.get("CASSIOPE Ephemeris/GEI Position (km)")
		self.gei_=np.array(gei_)

		gei_v=self.data.get("CASSIOPE Ephemeris/GEI Velocity (km per s)")
		self.gei_v=np.array(gei_v)

		gsm_=self.data.get("CASSIOPE Ephemeris/GSM Position (km)")
		self.gsm_=np.array(gsm_)

		glat_=self.data.get("CASSIOPE Ephemeris/Geographic Latitude (deg)")
		self.glat_=np.array(glat_).flatten()

		glon_=self.data.get("CASSIOPE Ephemeris/Geographic Longitude (deg)")
		self.glon_=np.array(glon_).flatten()

		MLT_=self.data.get("CASSIOPE Ephemeris/MLT (hr)")
		self.MLT_=np.array(MLT_).flatten()

		mlat_=self.data.get("CASSIOPE Ephemeris/Magnetic Latitude (deg)")
		self.mlat_=np.array(mlat_).flatten()

		mlon_=self.data.get("CASSIOPE Ephemeris/Magnetic Longitude (deg)")
		self.mlon_=np.array(mlon_).flatten()

		pitch_=self.data.get("CASSIOPE Ephemeris/Pitch (deg)")
		self.pitch_=np.array(pitch_).flatten()

		yaw_=self.data.get("CASSIOPE Ephemeris/Yaw (deg)")
		self.yaw_=np.array(yaw_).flatten()

		roll_=self.data.get("CASSIOPE Ephemeris/Roll (deg)")
		self.roll_=np.array(roll_).flatten()

		#calculate a unit vector (in GEI coordinates) for the spacecraft velocity
		self.epop_v=self.gei_v/np.sqrt(np.sum(np.square(self.gei_v),axis=0)) #e-POP X-vector, and also S/C velocity unit vector (in GEI)

		#calculate S/C unit z-vector is the negative of the displacement between the center of the GEI system and the S/C position
		#these are spacecraft centric unit vectors
		self.epop_z=-self.gei_/np.sqrt(np.sum(np.square(self.gei_),axis=0)) # e-POP Z-vector (in GEI)

		self.epop_y=np.cross(self.epop_z,self.epop_v,axisa=0,axisb=0,axisc=0) 	#the S/C unit y-vector is the Z cross S/C velocity

		self.epop_x=np.cross(self.epop_y,self.epop_z,axisa=0,axisb=0,axisc=0) #the S/C unit y-vector is the Z cross S/C velocity

		#RADIO DATA READ IN

		#Read in the monopole data, assign to numpy arrays
		#monopoles 1-4 mV
		m1_mV=self.data.get("RRI Data/Radio Data Monopole 1 (mV)") 
		self.m1_mV=np.array(m1_mV).flatten()
		
		m2_mV=self.data.get("RRI Data/Radio Data Monopole 2 (mV)") 
		self.m2_mV=np.array(m2_mV).flatten()

		m3_mV=self.data.get("RRI Data/Radio Data Monopole 3 (mV)") 
		self.m3_mV=np.array(m3_mV).flatten()

		m4_mV=self.data.get("RRI Data/Radio Data Monopole 4 (mV)") 
		self.m4_mV=np.array(m4_mV).flatten()

		#RRI's tuned frequency
		d1_freq=self.data.get("RRI Data/Channel A Frequencies (Hz)") #Dipole 1
		self.d1_freq=np.array(d1_freq)*40.000217171E6/40E6 #correction factor for RRI system clock
		
		d2_freq=self.data.get("RRI Data/Channel B Frequencies (Hz)") #Dipole 2
		self.d2_freq=np.array(d2_freq)*40.000217171E6/40E6 #correction factor for RRI system clock
		
		self.data.close() #close up the file since it's no longer in use



	# *** THE FOLLOWING METHOD HAS NOT BEEN VALIDATED YET **
	def RRI_point(self): #class method here to return the unit pointing vectors of each of RRI's boresight, for now, in GEI only

		#now do the rotations to account for the yaw pitch and roll of the spacecraft.
		Rx_=np.empty(shape=(3,3,len(self.roll_))) #empty roll matrix
		Ry_=np.empty(shape=(3,3,len(self.pitch_))) #empty pitch matrix
		Rz_=np.empty(shape=(3,3,len(self.yaw_))) #empty yaw matrix
		
		R_=np.empty(shape=(3,3,len(self.yaw_))) #empty rotation matrix
		
		RRI_point=np.empty(shape=(3,3,len(self.yaw_))) #empty RRI pointing matrix

		#build the roation matrices and RRI point
		for z in range(len(self.roll_)):
			
			Rx_[:,:,z]=Rotx(self.roll_[z])
			Ry_[:,:,z]=Roty(self.pitch_[z])
			Rz_[:,:,z]=Rotz(self.yaw_[z])	

			#RRI's pointing direction, accounting for the yaw, pitch, and roll
			#in RRI point, columnn 1 is the unit vector of the s/c x-vector, column 2 is the "" y-vector, column 3 "" z-vector
			#RRI point 3x3xtime
			RRI_point[:,:,z]=Rz_[:,:,z]@Ry_[:,:,z]@Rx_[:,:,z]@np.concatenate(([self.epop_x[:,z]],[self.epop_y[:,z]],[self.epop_z[:,z]]),axis=0).T 
			
		return RRI_point

		
	#this hasn't been validated yet

	def RRI_mono_point(self): #class method here to return the unit pointing vectors of each of RRI's monopoles, for now, in GEI only
	
		RRI_mono_point=np.empty(shape=(4,3,self.epop_x.shape[1])) #empty RRI monopole pointing matrix

		RRI_point_temp=self.RRI_point() #call RRI point, which gives you the pointing directions of all of e-POP's axes

		for z in range(self.epop_x.shape[1]):
			#since we know the x,y,z spacecraft vectors, the monopoles are simply vector additions of the spacecraft axis vectors
			RRI_mono_point[0,:,z]=(-RRI_point_temp[:,1,z]-RRI_point_temp[:,2,z])/np.sqrt(np.sum(np.square(-RRI_point_temp[:,1,z]-RRI_point_temp[:,2,z]))) # - y-z
			RRI_mono_point[1,:,z]=(RRI_point_temp[:,1,z]+RRI_point_temp[:,2,z])/np.sqrt(np.sum(np.square(RRI_point_temp[:,1,z]+RRI_point_temp[:,2,z]))) # y+z
			RRI_mono_point[2,:,z]=(RRI_point_temp[:,1,z]-RRI_point_temp[:,2,z])/np.sqrt(np.sum(np.square(RRI_point_temp[:,1,z]-RRI_point_temp[:,2,z]))) # y-z
			RRI_mono_point[3,:,z]=(-RRI_point_temp[:,1,z]+RRI_point_temp[:,2,z])/np.sqrt(np.sum(np.square(-RRI_point_temp[:,1,z]+RRI_point_temp[:,2,z]))) #-y+z

		return RRI_mono_point



	def MGF_data(self,fname_,start_dt,end_dt): #only for GEI.lv3 at this point in time, arguments are start and end datettime

		#class method here to read in and return e-POP MGF measurements
		#pass along time segement of interest in call
		#TO DO: include a time resolution request, should default to 1 sec

		#read in MGF level-3 ASCII file 
		f_=open(fname_,'r')
		mgf_dat=np.genfromtxt(f_,skip_header=1,usecols=(1,2,3,4)) #column 1 is e-POP MET, #2 is B_GEIX (nT), #3 B_GEIY (nT), #4 B_GEIZ (nT)
		f_.close()

		#MET epoch
		epop_epoch=datetime.datetime(1968,5,24,0,0,0)
		epop_epoch_s=time.mktime(epop_epoch.timetuple())#seconds since January 1, 1970 (Unix Epoch)

		#dealing with the time information
		mgf_s=np.array(mgf_dat[:,0])+epop_epoch_s-1 #MGF seconds in Unix epoch
		#MGF datettime object for this file in seconds resolution
		mgf_dt=datetime.datetime.fromtimestamp(mgf_s[0])+np.arange(len(mgf_s))*datetime.timedelta(seconds=1) 

		dt_id=np.where((mgf_dt<=end_dt)&(mgf_dt>=start_dt)) #narrowing down to the date of interest
		dt_id=dt_id[0] #dealing with the tuple

		#return MGF array with unit vector magnetic field (because I can only see those values mattering at this point)
		mgf_out=np.empty(shape=(3,len(dt_id)))
		mgf_out[0,:]=mgf_dat[dt_id,1]/np.sqrt(np.sum(np.square(mgf_dat[dt_id,1:4]),axis=1))
		mgf_out[1,:]=mgf_dat[dt_id,2]/np.sqrt(np.sum(np.square(mgf_dat[dt_id,1:4]),axis=1))
		mgf_out[2,:]=mgf_dat[dt_id,3]/np.sqrt(np.sum(np.square(mgf_dat[dt_id,1:4]),axis=1))

		return mgf_dt[dt_id],mgf_out #return the datetime and the MGF data in two separate arrays





#other class methods that are needed
#calculate Stokes
#calculate magnetic field vector (and unit vector) from MGF data file


