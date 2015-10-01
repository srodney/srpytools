"""
090915 S.Rodney
Crude Metropolis-Hastings
"""

def mcsample( p, x, Ndraws, x0=None, sigma=None, Nburnin=100, 
              debug=False) :
    """ p and x are arrays holding values that sample a
    probability distribution:  p(x).  
    We construct a Markov Chain with  Ndraws  steps using the 
    Metropolis-Hastings algorithm with a gaussian proposal distribution 
    of stddev sigma.  Default is sigma = (xmax-xmin)/10. 
    """
    from numpy import random
    from scipy import interpolate
    if debug: import pdb; pdb.set_trace()
    
    xmax = x[-1]
    xmin = x[0]
    if not sigma : sigma = (xmax-xmin)/10

    # define a linear interpolation probability function 
    plint = interpolate.interp1d( x, p, copy=False, 
                                  bounds_error=False, fill_value=0 )
    # if user doesn't provide a starting point, 
    # then draw an initial random position
    if not x0 : x0 = random.uniform( xmin, xmax )
    p0 = plint( x0 )  
    xsamples = []
    # for i in xrange(Ndraws) : 
    istep = 0
    while len(xsamples) < Ndraws :
        # draw a new position from a Gaussian proposal dist'n
        x1 = random.normal( x0, sigma ) 
        p1 = plint( x1 )
        # compare new against old position
        if p1>=p0 : 
            # new position has higher probability, so
            # accept it unconditionally
            if istep>Nburnin : xsamples.append( x1 ) 
            p0=p1
            x0=x1
        else : 
            # new position has lower probability, so 
            # pick new or old based on relative probs.
            y = random.uniform( )
            if y<p1/p0 :
                if istep>Nburnin : xsamples.append( x1 )
                p0=p1
                x0=x1
            else : 
                if istep>Nburnin : xsamples.append( x0 )
        istep +=1
    return( xsamples )


def sntest(sn, N=1000, Nbin=100, sigma=None, debug=False):
    from numpy import arange
    pz = sn.PDF.data1 * sn.PDF.XSTEP 
    pz = pz / pz.sum()
    zpk = sn.PDF.zpk
    nz = sn.PDF.NAXIS2
    dz = sn.PDF.XSTEP
    z0 = sn.PDF.XSTART
    z1 = sn.PDF.XSTART+nz*dz
    z = arange( z0, z1, dz )[:nz]
    
    if len(z) != len(pz): import pdb; pdb.set_trace()

    zsamples = mcsample( pz, z, N, sigma=sigma, debug=debug )

    from pylab import clf,plot,bar
    from numpy import histogram
    
    binct, binedge =  histogram(zsamples, Nbin, new=True, normed=False )
    binsize = binedge[1]-binedge[0]
    binct = binct.astype(float) /binct.sum()/binsize
    clf()
    bar(binedge[:-1], binct, binsize,color='slateblue' )
    plot( z, pz/pz.sum()/dz, ls='-', color='maroon', marker=' ',
          lw=3)
    #return( binct, binedge, z, pz )
