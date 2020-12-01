import matplotlib
matplotlib.use('pdf')
import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, vstack
from galpy.util import coords as transform
from galpy.util.bovy_conversion import time_in_Gyr
from matplotlib.patches import RegularPolygon
import astropy.coordinates as coord
import astropy.units as u
import matplotlib as mpl
from myspace import MySpace
from sklearn.mixture import GaussianMixture
from galpy.util import coords

def make_hexplot(XX,VV,gs=100):
    """
    Make set of 19 UV planes in hexagonal regions around the solar position
    Inputs: XX & VV are (N,3) position and velocity arrays
            gs=number of bins per axis
    (This is hacked together to centre the velocity distributions in the hexes)
    """
    coord = [[0,0,0],[0,400,-400],[-400,400,0],[-400,0,400],[0,-400,400],[400,-400,0],[400,0,-400],[0,800,-800],[-800,800,0],[-800,0,800],[0,-800,800],[800,-800,0],[800,0,-800],[800,0,0],[-400,800,-400],[-400,-800,400],[-800,0,0],[400,-800,400],[400,800,-400]]

    # Horizontal cartesian coords
    hcoord = [c[0] for c in coord]

    # Vertical cartersian coords
    vcoord = [2. * np.sin(np.radians(60)) * (c[1] - c[2]) /3. for c in coord]

    fig, ax = plt.subplots(1,figsize=(20,20))
    ax.set_aspect('equal')

    nhex=len(coord)
    # Add some coloured hexagons
    for i in range(0,nhex):
        xi=(XX[:,0])+hcoord[i]/1000.
        yi=(YY[:,1])+vcoord[i]/1000.
        idist=np.sqrt(xi**2+yi**2)
        iindx=(idist<0.2)*(np.fabs(XX[:,2])<0.2)
        hex = RegularPolygon((hcoord[i], vcoord[i]), numVertices=6, radius=800./3., 
                             orientation=np.radians(30), 
                             facecolor='white', alpha=1., edgecolor='k',zorder=i*2)
        ax.add_patch(hex)
        ax.hist2d(VV[:,0][iindx]*1.5+hcoord[i],VV[:,1][iindx]*2.-375.+220.+vcoord[i],range=[[hcoord[i]-220.,hcoord[i]+220],[vcoord[i]-220,vcoord[i]+220]],bins=gs,cmin=1.0e-50,rasterized=True,density=True,zorder=i*2+1)
        ax.set_xlim(-1200,1200)
        ax.set_ylim(-1200,1200)
        ax.set_xlabel(r'$x\ (\mathrm{pc})$',fontsize=20)
        ax.set_ylabel(r'$y\ (\mathrm{pc})$',fontsize=20)
        ax.tick_params(axis='both', labelsize=20)

    #hex2 = RegularPolygon((800,2.*np.sin(np.radians(60))*(1600) /3.), numVertices=6, radius=800./3.,orientation=np.radians(30), 
    #                      facecolor='white', alpha=1., edgecolor='k',zorder=0)
    #ax.add_patch(hex2)
    plt.savefig('Hexgrid.pdf',bbox_inches='tight')
    #plt.show()
    plt.close()

def make_before_after_animations(XX,VV,tensors,myspace,gs=150):
    """
    Make animations of X-Y, vx-vy, vx-vz, vy-vz in a series of wedges around the solar position beore and after transform
    Inputs: XX & VV are (N,3) position and velocity arrays
    gs=number of bins per axis
    tensors and myspace objects from MySpace
    """
    sr,sp,sz=coords.rect_to_cyl(XX[:,0], XX[:,1], XX[:,2])
    sp=sp+np.pi
    dist2=np.sqrt(XX[:,0]**2+XX[:,1]**2)
    rindx=(dist2>0.2)*(dist2<.5)*(np.fabs(XX[:,2])<0.2)
    for i in range(0,36):
        wedgedex=rindx*(sp>(i*np.pi/18.))*(sp<((i+3)*np.pi/18.))
        if i==34:
            wedgedex=rindx*(sp>(i*np.pi/18.))*(sp<((i+3)*np.pi/18.))+rindx*(sp>0.)*(sp<((1)*np.pi/18.))
        if i==35:
            wedgedex=rindx*(sp>(i*np.pi/18.))*(sp<((i+3)*np.pi/18.))+rindx*(sp>0.)*(sp<((2)*np.pi/18.))
        print(wedgedex.sum(),'stars in wedge',i)
                        
        f, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2, 2, figsize=(15,15))
        ax1.hist2d(XX[:,0][wedgedex],XX[:,1][wedgedex],range=[[-0.5,0.5],[-0.5,0.5]],bins=gs,cmin=1.0e-50,rasterized=True,density=True)
        ax1.set_xlabel(r'$X\ (\mathrm{kpc})$',fontsize=20)
        ax1.set_ylabel(r'$Y\ (\mathrm{kpc})$',fontsize=20)
        ax1.set_xlim(-0.5,0.5)
        ax1.set_ylim(-0.5,0.5)
        ax2.hist2d(VV[:,0][wedgedex],VV[:,1][wedgedex],range=[[-125,125],[-125,125]],bins=gs,cmin=1.0e-50,rasterized=True,density=True)
        ax2.set_xlabel(r'$v_X\ (\mathrm{km\ s}^{-1})$',fontsize=20)
        ax2.set_ylabel(r'$v_Y\ (\mathrm{km\ s}^{-1})$',fontsize=20)
        ax2.set_xlim(-125,125)
        ax2.set_ylim(-125,125)
        ax3.hist2d(VV[:,0][wedgedex],VV[:,2][wedgedex],range=[[-125,125],[-125,125]],bins=gs,cmin=1.0e-50,rasterized=True,density=True)
        ax3.set_xlabel(r'$v_X\ (\mathrm{km\ s}^{-1})$',fontsize=20)
        ax3.set_ylabel(r'$v_Z\ (\mathrm{km\ s}^{-1})$',fontsize=20)
        ax3.set_xlim(-125,125)
        ax3.set_ylim(-125,125)
        ax4.hist2d(VV[:,1][wedgedex],VV[:,2][wedgedex],range=[[-125,125],[-125,125]],bins=gs,cmin=1.0e-50,rasterized=True,density=True)
        ax4.set_xlabel(r'$v_Y\ (\mathrm{km\ s}^{-1})$',fontsize=20)
        ax4.set_ylabel(r'$v_Z\ (\mathrm{km\ s}^{-1})$',fontsize=20)
        ax4.set_xlim(-125,125)
        ax4.set_ylim(-125,125)
        ax1.tick_params(axis='both', which='major', labelsize=15)
        ax2.tick_params(axis='both', which='major', labelsize=15)
        ax3.tick_params(axis='both', which='major', labelsize=15)
        ax4.tick_params(axis='both', which='major', labelsize=15)
        plt.savefig('Pre-transform'+str("{:02d}".format(i))+'.pdf',bbox_inches='tight')
        plt.close()
    
        fix=myspace.get_model_v(tensors,VV[wedgedex],XX[wedgedex])
    
        f, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2, 2, figsize=(15,15))
        ax1.hist2d(XX[:,0][wedgedex],XX[:,1][wedgedex],range=[[-0.5,0.5],[-0.5,0.5]],bins=gs,cmin=1.0e-50,rasterized=True,density=True)
        ax1.set_xlabel(r'$X\ (\mathrm{kpc})$',fontsize=20)
        ax1.set_ylabel(r'$Y\ (\mathrm{kpc})$',fontsize=20)
        ax1.set_xlim(-0.5,0.5)
        ax1.set_ylim(-0.5,0.5)
        ax2.hist2d(fix[:,0],fix[:,1],range=[[-125,125],[-125,125]],bins=gs,cmin=1.0e-50,rasterized=True,density=True)
        ax2.set_xlabel(r'$v_X\ (\mathrm{km\ s}^{-1})$',fontsize=20)
        ax2.set_ylabel(r'$v_Y\ (\mathrm{km\ s}^{-1})$',fontsize=20)
        ax2.set_xlim(-125,125)
        ax2.set_ylim(-125,125)
        ax3.hist2d(fix[:,0],fix[:,2],range=[[-125,125],[-125,125]],bins=gs,cmin=1.0e-50,rasterized=True,density=True)
        ax3.set_xlabel(r'$v_X\ (\mathrm{km\ s}^{-1})$',fontsize=20)
        ax3.set_ylabel(r'$v_Z\ (\mathrm{km\ s}^{-1})$',fontsize=20)
        ax3.set_xlim(-125,125)
        ax3.set_ylim(-125,125)
        ax4.hist2d(fix[:,1],fix[:,2],range=[[-125,125],[-125,125]],bins=gs,cmin=1.0e-50,rasterized=True,density=True)
        ax4.set_xlabel(r'$v_Y\ (\mathrm{km\ s}^{-1})$',fontsize=20)
        ax4.set_ylabel(r'$v_Z\ (\mathrm{km\ s}^{-1})$',fontsize=20)
        ax4.set_xlim(-125,125)
        ax4.set_ylim(-125,125)
        ax1.tick_params(axis='both', which='major', labelsize=15)
        ax2.tick_params(axis='both', which='major', labelsize=15)
        ax3.tick_params(axis='both', which='major', labelsize=15)
        ax4.tick_params(axis='both', which='major', labelsize=15)
        plt.savefig('Post-transform'+str("{:02d}".format(i))+'.pdf',bbox_inches='tight')
        plt.close()

    os.system('convert -delay 5 -loop 0 Post-transform*.pdf After-sfbil.gif')
    os.system('convert -delay 5 -loop 0 Pre-transform*.pdf Before-sfbil.gif')

def make_corrections_animation(XX,VV,tensorsx,myspacex,tensorsxv,myspacexv,gs=150):
    """
    Make animation of X-Y & vx-vy in a series of wedges around the solar position for uncorrectd, x corrected and xv corrected velocity vectors
    Inputs: XX & VV are (N,3) position and velocity arrays
    gs=number of bins per axis
    tensorsx and myspacex objects from MySpace with only 'x'
    tensorsvx and myspace vx objects from MySpace with 'x' and 'xv'     
    """
    sr,sp,sz=bovy_coords.rect_to_cyl(XX[:,0], XX[:,1], XX[:,2])
    sp=sp+np.pi
    dist2=np.sqrt(XX[:,0]**2+XX[:,1]**2)
    rindx=(dist2>0.2)*(dist2<.5)*(np.fabs(XX[:,2])<0.2)
    for i in range(0,36):
        wedgedex=rindx*(sp>(i*np.pi/18.))*(sp<((i+3)*np.pi/18.))
        if i==34:
            wedgedex=rindx*(sp>(i*np.pi/18.))*(sp<((i+3)*np.pi/18.))+rindx*(sp>0.)*(sp<((1)*np.pi/18.))
        if i==35:
            wedgedex=rindx*(sp>(i*np.pi/18.))*(sp<((i+3)*np.pi/18.))+rindx*(sp>0.)*(sp<((2)*np.pi/18.))
        print(wedgedex.sum(),'stars in wedge',i)

        fixx=myspacex.get_model_v(tensorsx,VV[wedgedex],XX[wedgedex])
        fixxv=myspacexv.get_model_v(tensorsxv,VV[wedgedex],XX[wedgedex])

        f, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2, 2, figsize=(15,15))
        ax1.hist2d(XX[:,0][wedgedex],XX[:,1][wedgedex],range=[[-0.5,0.5],[-0.5,0.5]],bins=gs,cmin=1.0e-50,rasterized=True,density=True)
        ax1.set_xlabel(r'$X\ (\mathrm{kpc})$',fontsize=20)
        ax1.set_ylabel(r'$Y\ (\mathrm{kpc})$',fontsize=20)
        ax1.set_xlim(-0.5,0.5)
        ax1.set_ylim(-0.5,0.5)
        ax1.set_title(r'$\mathrm{Selected\ area}$',fontsize=20)
        ax2.hist2d(VV[:,0][wedgedex],VV[:,1][wedgedex],range=[[-125,125],[-125,125]],bins=gs,cmin=1.0e-50,rasterized=True,density=True)
        ax2.set_xlabel(r'$v_X\ (\mathrm{km\ s}^{-1})$',fontsize=20)
        ax2.set_ylabel(r'$v_Y\ (\mathrm{km\ s}^{-1})$',fontsize=20)
        ax2.set_xlim(-125,125)
        ax2.set_ylim(-125,125)
        ax2.set_title(r'$\mathrm{No\ correction}$',fontsize=20)
        ax3.hist2d(fixx[:,0],fixx[:,1],range=[[-125,125],[-125,125]],bins=gs,cmin=1.0e-50,rasterized=True,density=True)
        ax3.set_xlabel(r'$v_X\ (\mathrm{km\ s}^{-1})$',fontsize=20)
        ax3.set_ylabel(r'$v_Y\ (\mathrm{km\ s}^{-1})$',fontsize=20)
        ax3.set_xlim(-125,125)
        ax3.set_ylim(-125,125)
        ax3.set_title(r'$\mathrm{x\ correction}$',fontsize=20)
        ax4.hist2d(fixxv[:,0],fixxv[:,1],range=[[-125,125],[-125,125]],bins=gs,cmin=1.0e-50,rasterized=True,density=True)
        ax4.set_xlabel(r'$v_X\ (\mathrm{km\ s}^{-1})$',fontsize=20)
        ax4.set_ylabel(r'$v_Y\ (\mathrm{km\ s}^{-1})$',fontsize=20)
        ax4.set_xlim(-125,125)
        ax4.set_ylim(-125,125)
        ax4.set_title(r'$\mathrm{xv\ correction}$',fontsize=20)
        ax1.tick_params(axis='both', which='major', labelsize=15)
        ax2.tick_params(axis='both', which='major', labelsize=15)
        ax3.tick_params(axis='both', which='major', labelsize=15)
        ax4.tick_params(axis='both', which='major', labelsize=15)
        plt.savefig('orders'+str("{:02d}".format(i))+'.pdf',bbox_inches='tight')
        plt.close()

    os.system('convert -delay 5 -loop 0 orders*.pdf orders.gif')
