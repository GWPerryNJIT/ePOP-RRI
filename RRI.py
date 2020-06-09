#class created for e-POP RRI instrument 
#Created: May, 2020 by Gareth Perry

#import pdb; pdb.set_trace()


import numpy as np 
import h5py
import datetime, time 

fs=62500.33933 #RRI's sampling frequency


def Rotx(angle_): #funciton returning X-rotation matrix for a given angle
	Rx_=[[1,0,0],[0,np.cos(np.deg2rad(angle_)),-np.sin(np.deg2rad(angle_))],[0,np.sin(np.deg2rad(angle_)), np.cos(np.deg2rad(angle_))]]#	
	return Rx_

def Roty(angle_): #funciton returning Y-rotation matrix for a given angle
	Ry_=[[np.cos(np.deg2rad(angle_)),0,np.sin(np.deg2rad(angle_))],[0,1,0],[-np.sin(np.deg2rad(angle_)),0,np.cos(np.deg2rad(angle_))]]#	
	return Ry_

def Rotz(angle_): #funciton returning Z-rotation matrix for a given angle
	Rz_=[[np.cos(np.deg2rad(angle_)),-np.sin(np.deg2rad(angle_)),0],[np.sin(np.deg2rad(angle_)),np.cos(np.deg2rad(angle_)),0],[0,0,1]]#	
	return Rz_


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

		epop_s=self.data.get("CASSIOPE Ephemeris/Ephemeris MET (seconds since May 24, 1968)")
		epop_s=np.array(epop_s)+epop_epoch_s-1 #epop seconds in Unix epoch

		#ePOP datettime object for this file in seconds resolution
		self.epop_dt=datetime.datetime.fromtimestamp(epop_s[0])+np.arange(len(epop_s))*datetime.timedelta(seconds=1) 
		
		
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

		#calculate a unit vector (in GEI coordinates) for the spacecraft velocity
		epop_x=self.gei_v/np.sqrt(np.sum(np.square(self.gei_v),axis=0)) #e-POP X-vector, and also S/C velocity unit vector (in GEI)
		
		#calculate S/C unit z-vector is the negative of the displacement between the center of the GEI system and the S/C position
		epop_z=-self.gei_/np.sqrt(np.sum(np.square(self.gei_),axis=0)) # e-POP Z-vector (in GEI)
		
		epop_y=np.cross(epop_z,epop_x,axisa=0,axisb=0,axisc=0) 	#the S/C unit y-vector is the Z cross S/C velocity


		#now do the rotations to account for the yaw pitch and roll of the spacecraft.
		Rx_=np.empty(shape=(3,3,len(self.roll_))) #empty roll matrix
		Ry_=np.empty(shape=(3,3,len(self.pitch_))) #empty pitch matrix
		Rz_=np.empty(shape=(3,3,len(self.yaw_))) #empty yaw matrix
		
		R_=np.empty(shape=(3,3,len(self.yaw_))) #empty rotation matrix
		
		RRI_point=np.empty(shape=(3,len(self.yaw_))) #empty RRI pointing matrix

		#build the roation matrices and RRI point
		for z in range(len(self.roll_)):
			
			Rx_[:,:,z]=Rotx(self.roll_[z])
			Ry_[:,:,z]=Roty(self.pitch_[z])
			Rz_[:,:,z]=Rotz(self.yaw_[z])

			RxRy=np.matmul(Rx_[:,:,z],Ry_[:,:,z]) #doing this in two steps because I'm unnsure about the order of operations in matmul
			R_[:,:,z]=np.matmul(RxRy,Rz_[:,:,z]) #applying the rotations to form R_, the rotation matrix for all times

			RRI_point[:,z]=np.matmul(R_[:,:,z],epop_x[:,z]) #RRI's pointing direction, accounting for the yaw, pitch, and roll

		return RRI_point

		

	#this hasn't been scrutinized yet
	#*** WORKING ON THIS ***

	def RRI_mono_point(self): #class method here to return the unit pointing vectors of each of RRI's monopoles, for now, in GEI only

		RRI_point=self.RRI_point() #call the RRI_point method

		#determine which way the input rotation matrix is oriented
		mn_=RRI_point.shape

		if mn_[1]==3: #if the coorindate axis is axis=1 tranpose the matrix
			RRI_point=np.transpose(RRI_point)

		ll_=mn_[0]*mn_[1] #RRI pointing matrix
		
		RRI_mono_point=np.empty(shape=(4,3,ll_)) #empty RRI monopole pointing matrix

		#build the roation matrices for each monopole
		for z in range(ll_):
			R_1=np.matmul(Rotz(90),Rotx(-135)) #rotation matrix for RRI boresight to monopole 1
			R_2=np.matmul(Rotz(90),Rotx(45)) #rotation matrix for RRI boresight to monopole 2
			R_3=np.matmul(Rotz(90),Rotx(-45)) #rotation matrix for RRI boresight to monopole 3
			R_4=np.matmul(Rotz(90),Rotx(135)) #rotation matrix for RRI boresight to monopole 4


			#perform the rotations for each monopole
			RRI_mono_point[0,:,z]=np.matmul(RRI_point[:,z],R_1)
			RRI_mono_point[1,:,z]=np.matmul(RRI_point[:,z],R_2)
			RRI_mono_point[2,:,z]=np.matmul(RRI_point[:,z],R_3)
			RRI_mono_point[3,:,z]=np.matmul(RRI_point[:,z],R_4)		

		

		return RRI_monople_point

	

#other class methods that are needed
#calculate Stokes
#calculate magnetic field vector (and unit vector) from MGF data file
	#when doing this one, make sure everything is on the RRI 1-second time resolution

#fn1_=RRI('/Users/perry/Documents/Proposals/NASA/B13_US-Participating-Investigator_2020/RRI_20190601_004358_005056_lv1_v5.h5')

#print(fn1_.epop_dt)

