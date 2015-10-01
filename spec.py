# S.Rodney
# 2010.06.04
# working with 1-d fits files of spectra
from matplotlib import pyplot as pl
import numpy as np


def binspecfile( specdatfile, binwidth=10, wstart=0, wend=0 ):
    """ read in the spectrum from the given 
    spec.dat file, bin it up and return the binned
    wavelength and flux values.
    binwidth is in the wavelength units of the specdatfile
      (typically Angstroms)
    """
    import numpy
    try : 
        w,f,e = numpy.loadtxt( specdatfile, unpack=True )
    except:
        w,f = numpy.loadtxt( specdatfile, unpack=True )
        e = []

    wbinned, dw, fbinned, df  = binspecdat( w, f, e, binwidth=binwidth, wstart=wstart,wend=wend)
    return( wbinned, fbinned, df )


def binspecdat( wavelength, flux, fluxerr=[], binwidth=10, sigclip=0, sumerrs=False,
                wstart=0, wend=0 ):
    """  bin up the given wavelength and flux arrays
    and return the binned values.
    binwidth is in the wavelength units of the wavelength
    array  (typically Angstroms)
    """

    w,f = wavelength, flux
    wbinned, fbinned = [], []
    wbin,fbin,dfbin = np.array([]), np.array([]), np.array([])
    dw, df = [], []
    if wstart : istart = np.where( w>wstart )[0][0]
    else : istart = 0
    if wend : iend = np.where( w<wend )[0][-1]
    else : iend = len(w)
    w0 = w[istart]
    for i in range(istart,iend):
        fullbin = False
        if wend and w[i]>wend : break
        if w[i]>w0+binwidth :
            # determine the mean value in this bin
            w0 = w[i]
            igoodval = []
            if sigclip :
                # use sigma clipping to reject outliers
                igoodval = isigclip( fbin, sigclip )
                if len(igoodval) :
                    wbinval = np.mean( wbin[igoodval] )
                    fbinval = np.mean( fbin[igoodval] )
                    dwbinval = (wbin[igoodval].max() - wbin[igoodval].min())/2.
                    #dwbinval = (wbin.max() - wbin.min())/2.
                    if sumerrs :
                        # flux uncertainty is the quadratic sum of the mean flux error
                        # and the error of the mean
                        dfbinval1 = np.std( fbin[igoodval] ) / np.sqrt(len(igoodval)-2)
                        dfbinval2 = np.mean( dfbin[igoodval] ) / np.sqrt(len(igoodval)-2)
                        dfbinval = np.sqrt( dfbinval1**2 + dfbinval2**2 )
                    else :
                        # flux uncertainty is the std error of the mean
                        dfbinval = np.std( fbin[igoodval] ) / np.sqrt(len(igoodval)-2)

                    fullbin = True
                # note: if the binning is not successful, we continue building the bin
            else :
                # use a straight median
                wbinval = np.median( wbin )
                fbinval = np.median( fbin )
                dwbinval = (wbin[-1]-wbin[0])/2.
                if sumerrs :
                    # flux uncertainty is the quadratic sum of the mean flux error
                    # and the error of the mean
                    dfbinval1 = np.std( fbin )/np.sqrt(len(fbin)-2)
                    dfbinval2 = np.mean( dfbin )
                    dfbinval = np.sqrt( dfbinval1**2 + dfbinval2**2 )
                else :
                    # flux uncertainty is the std error of the mean
                    dfbinval = np.std( fbin ) / np.sqrt(max(1,len(fbin)))
                fullbin = True

            if fullbin :
                wbinned.append( wbinval )
                fbinned.append( fbinval )
                dw.append( dwbinval )
                df.append( dfbinval )

                # start a new bin
                wbin,fbin,dfbin = np.array([]), np.array([]), np.array([])

        # add a new data point to the bin
        wbin = np.append( wbin, w[i] )
        fbin = np.append( fbin, f[i] )
        if len(fluxerr):
            dfbin = np.append( dfbin, fluxerr[i] )
        else : dfbin = np.append( dfbin, 0 )

    return( np.array( wbinned ), np.array(dw), np.array(fbinned), np.array(df) )


def binspecdattomatch( wavelength, flux, wavetomatch, fluxerr=[],  sigclip=0,
                       sumerrs=False ):
    """  bin up the given wavelength and flux arrays
    and return the binned values.
    binwidth is in the wavelength units of the wavelength
    array  (typically Angstroms)
    """
    w,f = wavelength, flux
    if len(fluxerr):
        df = fluxerr
    else :
        df=np.zeros(len(f))

    wavetomatch = np.asarray(wavetomatch)
    wavetomatch_halfbinwidth = np.diff(wavetomatch)/2.
    lastbinlow = wavetomatch[-1] - wavetomatch_halfbinwidth[-1]
    lastbinhigh = wavetomatch[-1] + wavetomatch_halfbinwidth[-1]
    wavebinedges = np.append( wavetomatch[:-1]-wavetomatch_halfbinwidth,
                              np.array([lastbinlow,lastbinhigh]))

    wbinned, dwbinned, fbinned, dfbinned = [], [], [], []
    for i in range(len(wavebinedges)-1):
        wavebinmin=wavebinedges[i]
        wavebinmax=wavebinedges[i+1]
        iinbin = np.where((w>=wavebinmin)&(w<wavebinmax))

        winbin = w[iinbin]
        finbin = f[iinbin]
        dfinbin = df[iinbin]

        if sigclip :
            # use sigma clipping to reject outliers
            igoodval = isigclip( finbin, sigclip )
            if len(igoodval) :
                wbinval = np.mean( winbin[igoodval] )
                fbinval = np.mean( finbin[igoodval] )
                dwbinval = (winbin[igoodval].max() - winbin[igoodval].min())/2.
                #dwbinval = (wbin.max() - wbin.min())/2.
                if sumerrs :
                    # flux uncertainty is the quadratic sum of the mean flux error
                    # and the error of the mean
                    dfbinval1 = np.std( finbin[igoodval] ) / np.sqrt(len(igoodval)-2)
                    dfbinval2 = np.mean( dfinbin[igoodval] ) / np.sqrt(len(igoodval)-2)
                    dfbinval = np.sqrt( dfbinval1**2 + dfbinval2**2 )
                else :
                    # flux uncertainty is the std error of the mean
                    dfbinval = np.std( finbin[igoodval] ) / np.sqrt(len(igoodval)-2)
        else :
            # use a straight median
            wbinval = np.median( winbin )
            fbinval = np.median( finbin )
            dwbinval = (winbin[-1]-winbin[0])/2.
            if sumerrs :
                # flux uncertainty is the quadratic sum of the mean flux error
                # and the error of the mean
                dfbinval1 = np.std( finbin )/np.sqrt(len(finbin)-2)
                dfbinval2 = np.mean( dfinbin )
                dfbinval = np.sqrt( dfbinval1**2 + dfbinval2**2 )
            else :
                # flux uncertainty is the std error of the mean
                dfbinval = np.std( finbin ) / np.sqrt(max(1,len(finbin)))

        wbinned.append( wbinval )
        fbinned.append( fbinval )
        dwbinned.append( dwbinval )
        dfbinned.append( dfbinval )

    return( np.array( wbinned ), np.array(dwbinned), np.array(fbinned), np.array(dfbinned) )

def isigclip( valarray, sigclip, igood=[], maxiter=10, thisiter=0 ) : 
    """ find the indices of valarray that 
    survive after a clipping all values more than
    sigclip from the mean.  recursively iterative """
    if not type(valarray)==np.ndarray :
        valarray = np.array( valarray )
    if not len(igood) : igood = range(len(valarray))
    
    Ngood = len(igood)
    mnval = np.mean( valarray[igood] )
    sigma = np.std( valarray[igood] )
    igood = np.where( (np.abs(valarray-mnval)<(sigclip*sigma))  )[0]

    # import pdb; pdb.set_trace()
    if len(igood) == Ngood : return( igood )
    if thisiter>=maxiter :  
        print("WARNING : Stopping after %i recursions"%maxiter)
        return( igood )
    thisiter+=1
    igood = isigclip( valarray, sigclip, igood=igood, maxiter=maxiter, thisiter=thisiter )
    return( igood )


def rdspecfits( specfitsfile, ext='SCI', verbose=False ):
    """
    read in a 1-d spectrum from a fits file.  
    returns wavelength and flux as numpy arrays
    """
    from numpy import arange, zeros
    import exceptions
    import pyfits
    hdulist = pyfits.open( specfitsfile ) 

    try : 
        # reading a DEEP2/DEEP3 spectrum
        extroot='BXSPF'
        wb,fb,eb = hdulist['%s-B'%extroot].data[0][1], hdulist['%s-B'%extroot].data[0][0], hdulist['%s-B'%extroot].data[0][2]
        wr,fr,er = hdulist['%s-R'%extroot].data[0][1], hdulist['%s-R'%extroot].data[0][0], hdulist['%s-R'%extroot].data[0][2]
        return( np.append( wb, wr ), np.append( fb,fr ), np.append( eb,er) )
    except : 
        pass

    # determine the wavelength range 
    # covered by this spectrum 
    if len(hdulist) == 1 : ext = 0
    refwave = hdulist[ext].header['CRVAL1']
    refpix =  hdulist[ext].header['CRPIX1']
    if 'CD1_1' in hdulist[ext].header.keys() :
        dwave =   hdulist[ext].header['CD1_1']
    elif 'CDELT1' in hdulist[ext].header.keys() :
        dwave =   hdulist[ext].header['CDELT1']
    else :
        raise exceptions.RuntimeError(
            "wavelength step keyword not found")

    nwave =   hdulist[ext].header['NAXIS1']
    nap  =    hdulist[ext].header['NAXIS']
    widx = arange( nwave )
    wave = (widx - (refpix-1))*dwave + refwave
    flux = []
    if nap>1:
        for i in range( nap ):
            flux.append( hdulist[ext].data[i] )
    else : 
        flux = hdulist[ext].data
    return( wave, flux )


def specfits2dat( specfitsfile, specdatfile ):
    """ convert a 1-d spectrum fits file directly
    to a two-column ascii data text file """
    wave, flux = rdspecfits( specfitsfile )[:2]
    wspec2dat( wave, flux, specdatfile )
    return( specdatfile )


def wspec2dat( wave, flux, specdatfile, err=None):
    """ write wavelength and flux 
    into a two-column ascii data text file """
    fout = open( specdatfile, 'w' )
    if err is not None :
        for w,f,e in zip( wave, flux, err ):
            print >>fout, "%8.5e %8.5e %8.5e"%(w,f,e)
        fout.close()
    else :
        for w,f in zip( wave, flux ):
            print >>fout, "%8.5e %8.5e"%(w,f)
        fout.close()

def getSNIa( age=0, z=1 ):
    """ read in and return a SNIa spec file"""
    import os
    import numpy as np
    import sys

    thisfile = sys.argv[0]
    if 'ipython' in thisfile : thisfile = __file__
    thispath = os.path.abspath( os.path.dirname( thisfile ) )

    sedfile = os.path.join( thispath, 'Hsiao07.dat')
    d,w,f = np.loadtxt( sedfile, unpack=True ) 

    #d = d.astype(int)
    uniquedays = np.unique( d )
    ibestday = np.abs( uniquedays-age ).argmin()
    iday = np.where( d==uniquedays[ibestday] )

    dbest = d[ iday ]
    wbest = w[ iday ]
    fbest = f[ iday ]

    return( wbest*(1+z), fbest )

    

def matchSNIa( specfile, age=0, z=1.0, smooth=0, showerr=False): 
    """ interactive chi-by-eye SN spec fitting """
    pl.ion()
    pl.clf()

    snfile = getSNIa( age=age, z=z )
    ax1,ax2 = plotspecSNIa( specfile, age=age, z=z, smooth=smooth, showerr=showerr )
    ax1.set_xlim( 3500*(1+z), 9000*(1+z) )
    ax2.set_xlim( 3500, 9000 )
    userin=''
    print("age=%i z=%.3f"%(age, z))
    print("""
q : quit
h : help
z <zval> : set z
a <age> : set age in rest-frame days from peak
s <npix> : apply median smoothing, radius N pix (odd numbers only)
""")
    while userin!='q' :
        userin=raw_input('')
        if userin=='q': break
        elif userin.startswith('z') :
            z = float( userin.split()[1] )
            pl.clf()
            plotspecSNIa( specfile, age=age, z=z, smooth=smooth, showerr=showerr )
            pl.draw()
            print("re-plottd at z=%.3f"%z)
        elif userin.startswith('a') :
            age = int( userin.split()[1] )
            pl.clf()
            plotspecSNIa( specfile, age=age, z=z, smooth=smooth, showerr=showerr )
            pl.draw()
            print("re-plottd at age=%i"%age)
        elif userin.startswith('s') :
            smooth = int( userin.split()[1] )
            pl.clf()
            plotspecSNIa( specfile, age=age, z=z, smooth=smooth, showerr=showerr )
            print("re-plottd with smoothgin=%i"%smooth)
        elif userin=='h':
            print(""" 
q : quit
h : help
z <zval> : set z 
a <age> : set age in rest-frame days from peak
s <npix> : apply median smoothing, radius N pix (odd numbers only)
""")
    return( z )

    


def matchlines( specfile, skyfile=None, z=0.0, lineset='sdss', smooth=0, showerr=False):
    """Interactive line matching.
    h for help, q to quit"""
    pl.ion()
    pl.clf()
    plotspecsky( specfile, skyfile=skyfile, smooth=smooth )
    userin=''
    print( matchlines.__doc__ )
    print("z=%.3f %s"%(z,lineset))
    while userin!='q' : 
        userin=raw_input('')
        if userin=='q': break
        elif userin.startswith('z') :
            z = float( userin.split()[1] )
            pl.clf()
            plotspecsky( specfile, skyfile, smooth=smooth, showerr=showerr )
            if lineset!='none':marklines( z, lineset )
            pl.draw()
            print("re-plottd at z=%.3f"%z)
        elif userin.startswith('l') :
            lineset = userin.split()[1]
            pl.clf()
            plotspecsky( specfile, skyfile, smooth=smooth, showerr=showerr )
            if lineset != 'none' : marklines( z, lineset )
        elif userin.startswith('s') :
            smooth = int( userin.split()[1] )
            pl.clf()
            plotspecsky( specfile, skyfile, smooth=smooth, showerr=showerr )
            if lineset!='none': marklines( z, lineset )
        elif userin=='h':
            print(""" 
q : quit
h : help
z <zval> : set z 
l <linelist> : set line list [sdss,abs,em,sky,CuAr,none,SNIa]
s <npix> : apply median smoothing, radius N pix
""")
    return( z )

def plotspecsky( specfile, skyfile=None, smooth=0, showerr=False ):
    """ plot the source spectrum and the sky spectrum """
    # medsmooth = lambda f,N : array( [ median( f[max(0,i-N):min(len(f),max(0,i-N)+2*N)]) for i in range(len(f)) ] )

    if skyfile : 
        # lower axes : sky
        ax2 = pl.axes([0.03,0.05,0.95,0.2])
        skywave, skyflux = np.loadtxt( skyfile, unpack=True, usecols=[0,1] )
        pl.plot( skywave, skyflux , color='darkgreen',
              ls='-', drawstyle='steps' )
        ax1 = pl.axes([0.03,0.25,0.95,0.63], sharex=ax2)

    # upper axes : source 
    # TODO : better smoothing !!!
    try : 
        wave, flux, fluxerr = np.loadtxt( specfile, unpack=True, usecols=[0,1,2] )
    except : 
        wave, flux = np.loadtxt( specfile, unpack=True, usecols=[0,1] )
        fluxerr = np.zeros( len(flux) )
    # if smooth : flux = medsmooth( flux, smooth )
    if smooth : 
        if smooth<5 : 
            smooth=5
            order=3
            print("raising S-G smooth window to 5, order 3.")
        if smooth<7 : 
            order=3
        else : 
            order=5
        flux = savitzky_golay( flux, smooth, order=order )
    if showerr : 
        pl.errorbar( wave, flux/np.median(flux), fluxerr/np.median(flux), marker=' ', color='k', ls='-', drawstyle='steps' )
    else : 
        pl.plot( wave, flux/np.median(flux), marker=' ', color='k', ls='-', drawstyle='steps' )
    #draw()
    #show()
    return()

def plotspecSNIa( specfile, age=0, z=1, smooth=0, showerr=False, scale=0, color='k' ):
    """ plot the source spectrum and overlay a SN spectrum """
    import numpy as np
    from scipy import interpolate as scint
    medsmooth = lambda f,N : np.array( [ np.median( f[max(0,i-N):min(len(f),max(0,i-N)+2*N)]) for i in range(len(f)) ] )

    try : 
        wave, flux, fluxerr = np.loadtxt( specfile, unpack=True, usecols=[0,1,2] )
        # fluxerr = fluxerr / 5.
    except : 
        wave, flux = np.loadtxt( specfile, unpack=True, usecols=[0,1] )
        fluxerr = np.ones( len(flux) )
        showerr=False
    
    snwave, snflux = getSNIa( age, z )

    snfinterp = scint.interp1d( snwave, snflux, bounds_error=False, fill_value=0 ) 
    snf = snfinterp( wave )

    if smooth>0 :
        if smooth<5 : 
            smooth=5
            order=3
            print("raising S-G smooth window to 5, order 3.")
        if smooth<7 : 
            order=3
        else : 
            order=5
        flux = savitzky_golay( flux, smooth, order=order )
    elif smooth<0:
        flux = medsmooth( flux, abs(smooth) )


    if scale :
        if showerr :
            pl.errorbar( wave, flux, fluxerr, marker=' ', color=color, ls='-', drawstyle='steps-mid', capsize=0, lw=0.5, scalex=False )
        else :
            pl.plot( wave, flux, marker=' ', color=color, ls='-', drawstyle='steps', lw=0.5, ) # , scalex=False )
        pl.plot( snwave, snflux * scale , marker=' ', color='r', ls='-', lw=1.5, scalex=False )

    else :
        num = np.sum(  snf*flux / fluxerr**2 )
        denom = np.sum( snf**2./ fluxerr**2 )
        scale = num / denom

        if showerr :
            pl.errorbar( wave, flux/np.median(flux), fluxerr/np.median(flux), marker=' ', color=color, ls='-', drawstyle='steps-mid', capsize=0, lw=0.5, scalex=False )
        else :
            pl.plot( wave, flux/np.median(flux), marker=' ', color=color, ls='-', drawstyle='steps', lw=0.5, ) # , scalex=False )
        pl.plot( snwave, snflux * scale / np.median(flux) , marker=' ', color='r', ls='-', lw=1.5, scalex=False )

    ax1 = pl.gca()
    ax2 = ax1.twiny()
    ax2.set_xlim( ax1.get_xlim()[0] / (1+z), ax1.get_xlim()[1] / (1+z) )
    ax1.set_xlabel('Observed Wavelength (\AA)')
    ax2.set_xlabel('Rest Wavelength (\AA)')

    return(ax1,ax2)



def savitzky_golay(y, window_size=5, order=3, deriv=0):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techhniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError, msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv]
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m, y, mode='valid')


def ccm_unred(wave, flux, ebv, r_v=""):
    """ccm_unred(wave, flux, ebv, r_v="")
    Deredden a flux vector using the CCM 1989 parameterization 
    Returns an array of the unreddened flux
  
    INPUTS:
    wave - array of wavelengths (in Angstroms)
    dec - calibrated flux array, same number of elements as wave
    ebv - colour excess E(B-V) float. If a negative ebv is supplied
          fluxes will be reddened rather than dereddened     
  
    OPTIONAL INPUT:
    r_v - float specifying the ratio of total selective
          extinction R(V) = A(V)/E(B-V). If not specified,
          then r_v = 3.1
  
    OUTPUTS:
    funred - unreddened calibrated flux array, same number of 
             elements as wave
  
    NOTES:
    1. This function was converted from the IDL Astrolib procedure
       last updated in April 1998. All notes from that function
       (provided below) are relevant to this function 
  
    2. (From IDL:) The CCM curve shows good agreement with the Savage & Mathis (1979)
       ultraviolet curve shortward of 1400 A, but is probably
       preferable between 1200 and 1400 A.
  
    3. (From IDL:) Many sightlines with peculiar ultraviolet interstellar extinction 
       can be represented with a CCM curve, if the proper value of 
       R(V) is supplied.
  
    4. (From IDL:) Curve is extrapolated between 912 and 1000 A as suggested by
       Longo et al. (1989, ApJ, 339,474)
  
    5. (From IDL:) Use the 4 parameter calling sequence if you wish to save the 
       original flux vector.
  
    6. (From IDL:) Valencic et al. (2004, ApJ, 616, 912) revise the ultraviolet CCM
       curve (3.3 -- 8.0 um-1).    But since their revised curve does
       not connect smoothly with longer and shorter wavelengths, it is
       not included here.
  
    7. For the optical/NIR transformation, the coefficients from 
       O'Donnell (1994) are used
  
    >>> ccm_unred([1000, 2000, 3000], [1, 1, 1], 2 ) 
    array([9.7976e+012, 1.12064e+07, 32287.1])
    """
    import numpy as np
    wave = np.array(wave, float)
    flux = np.array(flux, float)
  
    if wave.size != flux.size: raise TypeError, 'ERROR - wave and flux vectors must be the same size'
  
    if not bool(r_v): r_v = 3.1
  
    x = 10000.0/wave
    npts = wave.size
    a = np.zeros(npts, float)
    b = np.zeros(npts, float)
  
    ###############################
    #Infrared
  
    good = np.where( (x > 0.3) & (x < 1.1) )
    a[good] = 0.574 * x[good]**(1.61)
    b[good] = -0.527 * x[good]**(1.61)
  
    ###############################
    # Optical & Near IR
  
    good = np.where( (x  >= 1.1) & (x < 3.3) )
    y = x[good] - 1.82
  
    c1 = np.array([ 1.0 , 0.104,   -0.609,    0.701,  1.137, \
                  -1.718,   -0.827,    1.647, -0.505 ])
    c2 = np.array([ 0.0,  1.952,    2.908,   -3.989, -7.985, \
                  11.102,    5.491,  -10.805,  3.347 ] )
  
    a[good] = np.polyval(c1[::-1], y)
    b[good] = np.polyval(c2[::-1], y)
  
    ###############################
    # Mid-UV
  
    good = np.where( (x >= 3.3) & (x < 8) )
    y = x[good]
    F_a = np.zeros(np.size(good),float)
    F_b = np.zeros(np.size(good),float)
    good1 = np.where( y > 5.9 )
  
    if np.size(good1) > 0:
        y1 = y[good1] - 5.9
        F_a[ good1] = -0.04473 * y1**2 - 0.009779 * y1**3
        F_b[ good1] =   0.2130 * y1**2  +  0.1207 * y1**3
  
    a[good] =  1.752 - 0.316*y - (0.104 / ( (y-4.67)**2 + 0.341 )) + F_a
    b[good] = -3.090 + 1.825*y + (1.206 / ( (y-4.62)**2 + 0.263 )) + F_b
  
    ###############################
    # Far-UV
  
    good = np.where( (x >= 8) & (x <= 11) )
    y = x[good] - 8.0
    c1 = [ -1.073, -0.628,  0.137, -0.070 ]
    c2 = [ 13.670,  4.257, -0.420,  0.374 ]
    a[good] = np.polyval(c1[::-1], y)
    b[good] = np.polyval(c2[::-1], y)
  
    # Applying Extinction Correction
  
    a_v = r_v * ebv
    a_lambda = a_v * (a + b/r_v)
  
    funred = flux * 10.0**(0.4*a_lambda)   
  
    return funred


    
def marklines( z, lineset, telluric=True ):
    import rcpar
    #rcpar.usetex()

    if lineset=='sdss' : lineset = sdsslines
    elif lineset=='sn' : lineset = snialines
    elif lineset=='ia' : lineset = snialines
    elif lineset=='abs' : lineset = abslines
    elif lineset=='agn' : lineset = agnlines
    elif lineset=='em' : lineset = emissionlines
    elif lineset=='sky' : lineset = skylines
    elif lineset=='CuAr' : lineset = CuArlines

    ax = pl.gca()
    ax.set_autoscale_on(False)
    ymin = ax.get_ylim()[0]
    ymax = ax.get_ylim()[1]
    y = 0.8
    i=0
    for linepair in lineset : 
        if i==0: c,y='r',0.8
        elif i==1: c,y='g',0.75
        elif i==2: c,y='b',0.70
        i+=1
        if i>2: i=0
        w = linepair[0]
        name = linepair[1]
        ax.axvline( w * (1+z), color=c )
        ax.text( w * (1+z)+10, y*(ymax-ymin)+ymin, name, color=c)
    if telluric : 
        ax.bar(7589,2*(ymax-ymin), bottom=ymin-ymax,
            width=70, alpha=0.1, color='k' )
        ax.text( 7593, ymin+(ymax-ymin)/2, r'$\bigoplus$',color='k')
    #show()
    return()

agnlines = [
[1033.30004882812, 'OVI'],
[1215.67004394531, r'Ly$\alpha$'],
[1239.42004394531, 'NV'],
[1305.53002929688, 'OI'],
[1335.52001953125, 'CII'],
[1399.80004882812, 'SiIV,OIV'],
[1545.85998535156, 'CIV'],
[1640.40002441406, 'HeII'],
[1665.84997558594, 'OIII'],
[1857.40002441406, 'AlIII'],
[1908.27001953125, 'CIII'],
[2326.0, 'CII'],
[2439.5, 'NeIV'],
[2800.32006835938, 'MgII'],
[3346.7900390625, 'NeV'],
[3426.85009765625, 'NeV'],
[3728.30004882812, 'OII'],
[3798.97607421875, 'H$\theta$'],
[3836.46997070312, 'He'],
[3889.0, 'HeI'],
[4072.30004882812, 'SII'],
[4102.89013671875, 'Hd'],
[4341.68017578125, 'Hg'],
[4364.43603515625, 'OIII'],
[4686, 'HeII'],
[4862.68017578125, r'H$\beta$'],
[4960.294921875, 'OIII'],
[5008.240234375, 'OIII'],
[6302.0458984375, 'OI'],
[6365.5361328125, ''],
[6549.85986328125, ''],
[6564.60986328125, ''],
[6585.27001953125, r'H$\alpha$,NII'],
[6718.2900390625, ''],
[6732.669921875, 'SII'],
]

snialines = [
[3850,'CaII H&K'],
[4000,'SiII'],
[4300,'MgII'],
[4800,'FeII'],
[5400,'SiII-W'],
[5800,'SiII'],
[6150,'SiII 6150'],
[8100,'CaII IR'],
]

abslines = [
[3241.98388671875, 'TI_II'],
[3302.36889648438, 'NA_I'],
[3302.97900390625, 'NA_I'],
[3383.76098632812, 'TI_II'],
[3933.6630859375, 'CA_II'],
[3968.46801757812, 'CA_II'],
[4102.89013671875, 'H_delta'],
[4226.72802734375, 'CA_I'],
[4276.830078125, '[FeII]'],
[4287.39990234375, '[FeII]'],
[4341.68017578125, 'H_gamma'],
[4364.43603515625, 'OIII'],
[4686.0, 'HeII4686'],
[4862.68017578125, 'H_beta'],
[5176.7001953125, 'Mg'],
[5876.0, '[HeI]5876'],
[5889.9501953125, 'NA_I'],
[5895.923828125, 'NA_I'],
[6564.60986328125, 'H_alpha'],
[7664.89892578125, 'K_I'],
[7698.958984375, 'K_I'],
]

sdsslines = [
[1033.30004882812, 'OVI'],
[1215.67004394531, 'Ly_alpha'],
[1239.42004394531, 'NV'],
[1305.53002929688, 'OI'],
[1335.52001953125, 'CII'],
[1399.80004882812, 'SiIV+OIV'],
[1545.85998535156, 'CIV'],
[1640.40002441406, 'HeII'],
[1665.84997558594, 'OIII'],
[1857.40002441406, 'AlIII'],
[1908.27001953125, 'CIII'],
[2326.0, 'CII'],
[2439.5, 'NeIV'],
[2800.32006835938, 'MgII'],
[3346.7900390625, 'NeV'],
[3426.85009765625, 'NeV'],
[3728.30004882812, 'OII'],
[3798.97607421875, 'H_theta'],
[3836.46997070312, 'H_eta'],
[3889.0, 'HeI'],
[3934.77709960938, 'K'],
[3969.587890625, 'H'],
[4072.30004882812, 'SII'],
[4102.89013671875, 'H_delta'],
[4305.60986328125, 'G'],
[4341.68017578125, 'H_gamma'],
[4364.43603515625, 'OIII'],
[4862.68017578125, 'H_beta'],
[4960.294921875, 'OIII'],
[5008.240234375, 'OIII'],
[5176.7001953125, 'Mg'],
[5895.60009765625, 'Na'],
[6302.0458984375, 'OI'],
[6365.5361328125, 'OI'],
[6549.85986328125, 'NII'],
[6564.60986328125, 'H_alpha'],
[6585.27001953125, 'NII'],
[6707.89013671875, 'Li'],
[6718.2900390625, 'SII'],
[6732.669921875, 'SII'],
]

emissionlines = [                 
[3726.15991210938, '[OII]'],
[3728.90991210938, '[OII]'],
[4101.0, 'HDELTA'],
[4276.830078125, '[FeII]'],
[4287.39990234375, '[FeII]'],
[4319.6201171875, '[FeII]'],
[4340.4677734375, 'HGAMMA'],
[4363.2099609375, '[OIII]4363'],
[4413.77978515625, '[FeII]'],
[4416.27001953125, '[FeII]'],
[4686.0, 'HeII4686'],
[4861.31982421875, 'HBETA'],
[4889.6201171875, '[FeII]'],
[4905.33984375, '[FeII]'],
[4958.91015625, '[OIII]4959'],
[5006.83984375, '[OIII]5007'],
[5111.6298828125, '[FeII]'],
[5158.77978515625, '[FeII]'],
[5199.60009765625, '[NI]'],
[5261.6201171875, '[FeII]'],
[5577.31005859375, '[OI]'],
[5876.0, '[HeI]5876'],
[6300.2998046875, '[OI]'],
[6363.77978515625, '[OI]'],
[6548.10009765625, '[NII]6548'],
[6562.81689453125, 'HALPHA'],
[6583.60009765625, '[NII]6584'],
[6716.47021484375, '[SII]6717'],
[6730.85009765625, '[SII]6731'],
[7002.0, 'OI'],
[7005.7001953125, '[ArV]'],
[7065.2998046875, 'HeI'],
[7135.7998046875, '[ArIII]'],
[7236.2001953125, 'CII'],
[7254.39990234375, 'OI'],
[7281.2998046875, 'HeI'],
[7291.4599609375, '[CaII]'],
[7319.60009765625, '[OII]'],
[7323.8798828125, '[CaII]'],
[7330.2001953125, '[OIII]'],
[7377.7998046875, '[NiII]'],
[7452.5, '[FeII]'],
[7468.2998046875, 'NI'],
[7751.10009765625, '[ArIII]'],
[7816.2001953125, 'HeI'],
[7889.89990234375, '[NiIII]'],
[8392.400390625, 'HI-Pa20'],
[8413.2998046875, 'HI-Pa19'],
[8438.0, 'HI-Pa18'],
[8446.400390625, 'OI'],
[8467.2998046875, 'HI-Pa17'],
[8502.5, 'HI-Pa16'],
[8545.400390625, 'HI-Pa15'],
[8578.7001953125, '[ClII]'],
[8598.400390625, 'HI-Pa14'],
[8617.0, '[FeII]'],
[8665.0, 'HI-Pa13'],
[8681.599609375, 'NI'],
[8703.2001953125, 'NI'],
[8711.7001953125, 'NI'],
[8750.5, 'HI-Pa12'],
[8862.7998046875, 'HI-Pa11'],
[9014.900390625, 'HI-Pa10'],
[9069.0, '[SIII]'],
[9229.0, 'HI-Pa9'],
[9530.900390625, '[SIII]'],
[9546.0, 'HI-Pa8'],
[9824.099609375, '[CI]'],
[9850.2998046875, '[CI]'],
[10031.2001953125, 'HeI'],
[10049.400390625, 'HI-Pa7'],
[10286.7001953125, '[SII]'],
[10320.5, '[SII]'],
[10336.400390625, '[SII]'],
[10830.2001953125, 'HeI'],
[10938.099609375, 'HI-Pa6'],]

skylines = [
[1033.30004882812, 'OVI'],
[1215.67004394531, 'Ly_alpha'],
[1239.42004394531, 'NV'],
[1305.53002929688, 'OI'],
[1335.52001953125, 'CII'],
[1399.80004882812, 'SiIV+OIV'],
[1545.85998535156, 'CIV'],
[1640.40002441406, 'HeII'],
[1665.84997558594, 'OIII'],
[1857.40002441406, 'AlIII'],
[1908.27001953125, 'CIII'],
[2326.0, 'CII'],
[2439.5, 'NeIV'],
[2800.32006835938, 'MgII'],
[3346.7900390625, 'NeV'],
[3426.85009765625, 'NeV'],
[3728.30004882812, 'OII'],
[3798.97607421875, 'H_theta'],
[3836.46997070312, 'H_eta'],
[3889.0, 'HeI'],
[3934.77709960938, 'K'],
[3969.587890625, 'H'],
[4072.30004882812, 'SII'],
[4102.89013671875, 'H_delta'],
[4305.60986328125, 'G'],
[4341.68017578125, 'H_gamma'],
[4364.43603515625, 'OIII'],
[4862.68017578125, 'H_beta'],
[4960.294921875, 'OIII'],
[5008.240234375, 'OIII'],
[5176.7001953125, 'Mg'],
[5895.60009765625, 'Na'],
[6302.0458984375, 'OI'],
[6365.5361328125, 'OI'],
[6549.85986328125, 'NII'],
[6564.60986328125, 'H_alpha'],
[6585.27001953125, 'NII'],
[6707.89013671875, 'Li'],
[6718.2900390625, 'SII'],
[6732.669921875, 'SII'],
]

skylines2= [
[4047.0, "Hg-Ltg*"],
[4358.0, "Hg-Ltg*"],
[4669.0, "NaI_HPS_ltg*"],
[4983.0, "NaI_HPS_ltg*"],
[5149.0, "NaI_HPS_ltg*"],
[5199.0, "[NI]*"],
[5460.0, "Hg-Ltg*"],
[5577.31005859375, "[OI]"],
[5604.60009765625, "OH(7,1)-P1(3)*"],
[5622.0, "OH(7,1)-P1(4)*"],
[5641.7998046875, "OH(7,1)-P1(5)*"],
[5665.5, "OH(7,1)-P1(6)*"],
[5668.0, "NaI_HPS_ltg*"],
[5690.7001953125, "OH(7,1)-P1(7)*"],
[5718.7998046875, "OH(7,1)-P1(8)*"],
[5770.0, "Hg-Ltg*"],
[5791.0, "Hg-Ltg*"],
[5890.0, "Hg-Ltg*"],
[5893.0, "NaI5893*"],
[5896.0, "Hg-Ltg*"],
[5905.5, "OH(8,2)-P2(1)*"],
[5914.2998046875, "OH(8,2)-P1(2)*"],
[5923.5, "OH(8,2)-P2(3)*"],
[5932.0, "OH(8,2)-P1(3)*"],
[5944.10009765625, "OH(8,2)-P2(4)*"],
[5953.10009765625, "OH(8,2)-P1(4)*"],
[5968.5, "OH(8,2)-P2(5)*"],
[5975.10009765625, "OH(8,2)-P1(5)*"],
[5995.39990234375, "OH(8,2)-P2(6)*"],
[6002.2998046875, "OH(8,2)-P1(6)*"],
[6032.0, "OH(8,2)-P1(7)*"],
[6065.2998046875, "OH(8,2)-P1(8)*"],
[6161.0, "NaI_HPS_ltg*"],
[6192.0, "OH(5,0)-P2(2)*"],
[6200.7001953125, "OH(5,0)-P1(2)*"],
[6211.89990234375, "OH(5,0)-P2(3)*"],
[6220.5, "OH(5,0)-P1(3)*"],
[6258.60009765625, "OH(9,3)-Q(1)*"],
[6265.10009765625, "OH(9,3)-Q(2)*"],
[6287.2001953125, "OH(9,3)-P1(2)*"],
[6300.2998046875, "[OI]"],
[6307.2001953125, "OH(9,3)-P1(3)*"],
[6321.2998046875, "OH(9,3)-P2(4)*"],
[6329.39990234375, "OH(9,3)-P1(4)*"],
[6349.0, "OH(9,3)-P2(5)*"],
[6355.5, "OH(9,3)-P1(5)*"],
[6363.77978515625, "[OI]"],
[6379.60009765625, "OH(9,3)-P2(6)*"],
[6386.2998046875, "OH(9,3)-P1(6)*"],
[6415.60009765625, "OH(9,3)-P2(7)*"],
[6421.2001953125, "OH(9,3)-P1(7)*"],
[6452.5, "OH(9,3)-P2(8)*"],
[6455.5, "OH(9,3)-P1(8)*"],
[6499.06005859375, "OH(6,1)-Q(1)"],
[6504.81005859375, "OH(6,1)-Q(2)"],
[6513.669921875, "OH(6,1)-Q(3)"],
[6522.0400390625, "OH(6,1)-P2(2)"],
[6532.77978515625, "OH(6,1)-P1(2)"],
[6544.2001953125, "OH(6,1)-P2(3)"],
[6553.3798828125, "OH(6,1)-P1(3)"],
[6562.81689453125, "HALPHA"],
[6568.7099609375, "OH(6,1)-P2(4)"],
[6577.009765625, "OH(6,1)-P1(4)"],
[6596.52978515625, "OH(6,1)-P2(5)"],
[6603.759765625, "OH(6,1)-P1(5)"],
[6627.7099609375, "OH(6,1)-P2(6)"],
[6634.31005859375, "OH(6,1)-P1(6)"],
]


#------------------------------------------------------------
# CuAr Line List: 3094A-10470A
# units Angstroms
CuArlines = [
[3478.2324, ''],
[3491.2440, ''],
[3561.0304, ''],
[3718.2065, ''],
[3729.3087, ''],
[3737.889, ''],
[3765.27, ''],
[3850.5813, ''],
[3868.5284, ''],
[3891.9792, ''],
[3946.0971, ''],
[3994.7918, ''],
[4013.8566, ''],
[4033.8093, ''],
[4044.4179, ''],
[4052.9208, ''],
[4072.3849, ''],
[4079.5738, ''],
[4103.9121, ''],
[4131.7235, ''],
[4158.5905, ''],
[4181.8836, ''],
[4191.0294, ''],
[4199.8891, ''],
[4228.158, ''],
[4237.2198, ''],
[4259.3619, ''],
[4277.5282, ''],
[4300.1008, ''],
[4309.2392, ''],
[4333.5612, ''],
[4348.064, ''],
[4370.7532, ''],
[4400.9863, ''],
[4448.8792, ''],
[4474.7594, ''],
[4481.8107, ''],
[4510.7332, ''],
[4545.0519, ''],
[4563.7429, ''],
[4579.3495, ''],
[4589.8978, ''],
[4596.0967, ''],
[4598.7627, ''],
[4609.5673, ''],
[4628.4409, ''],
[4637.2328, ''],
[4657.9012, ''],
[4702.3161, ''],
[4726.8683, ''],
[4764.8646, ''],
[4806.0205, ''],
[4847.8095, ''],
[4865.9105, ''],
[4879.8635, ''],
[4889.0422, ''],
[4904.7516, ''],
[4933.2091, ''],
[4965.0795, ''],
[5009.3344, ''],
[5017.1628, ''],
[5062.0371, ''],
[5090.4951, ''],
[5151.3907, ''],
[5162.2846, ''],
[5187.7462, ''],
[5421.3517, ''],
[5451.6520, ''],
[5495.8738, ''],
[5506.1128, ''],
[5558.7020, ''],
[5572.5413, ''],
[5606.7330, ''],
[5650.7043, ''],
[5739.5196, ''],
[5772.1143, ''],
[5802.0798, ''],
[5834.2633, ''],
[5860.3103, ''],
[5888.5841, ''],
[5912.0853, ''],
[5928.8130, ''],
[6032.1274, ''],
[6043.2233, ''],
[6059.3725, ''],
[6114.9234, ''],
[6145.4411, ''],
[6155.2385, ''],
[6172.2780, ''],
[6296.8722, ''],
[6307.6570, ''],
[6364.8937, ''],
[6369.5748, ''],
[6384.7169, ''],
[6416.3071, ''],
[6466.5526, ''],
[6483.0825, ''],
[6677.2817, ''],
[6752.8335, ''],
[6871.2891, ''],
[6937.6642, ''],
[6965.4307, ''],
[7030.2514, ''],
[7067.2181, ''],
[7147.0416, ''],
[7206.9804, ''],
[7272.9359, ''],
[7311.7159, ''],
[7353.2930, ''],
[7383.9805, ''],
[7503.8691, ''],
[7635.1060, ''],
[7723.7611, ''],
[7948.1764, ''],
[8103.6931, ''],
[8115.311, ''],
[8264.5225, ''],
[8408.2096, ''],
[8424.6475, ''],
[8667.9442, ''],
[9122.9674, ''],
[9224.4992, ''],
[9354.2198, ''],
[9657.7863, ''],
[9784.5028, ''],
[10470.0535, ''] ]




