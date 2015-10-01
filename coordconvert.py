#! /usr/bin/env python
"""
2012.04.15 S.Rodney
Convert RA,Dec coordinates from decimal to sexagesimal or vice versa

Examples:  

  coordconvert 03:32:38.01 -27:46:39.08
53.158375   -27.777522

 - OR - 

  coordconvert 34.442967 -5.256581
02:17:46.312   -05:15:23.69 

"""

def main() :
    import sys
    import exceptions

    Narg = len(sys.argv) - 1 
    if not Narg : 
        print __doc__
        sys.exit()

    ra = sys.argv[1]
    dec = sys.argv[2] 

    try : 
        convertAstropy( ra, dec )
    except ImportError : 
        try: 
            convertAstLib( ra, dec )
        except ImportError : 
            raise exceptions.ImportError( "No astropy and no astLib! Install one of those and try again.")


def convertAstLib( ra, dec ) : 
    from astLib import astCoords
    if ':' in ra : 
        raout = '%.6f' % astCoords.hms2decimal( ra, delimiter=':' )
        decout = '%.6f' %  astCoords.dms2decimal( dec, delimiter=':' )
    else : 
        raout = astCoords.decimal2hms( float(ra), delimiter=':' )
        decout = astCoords.decimal2dms( float(dec), delimiter=':' )
    print( raout + '   ' + decout )
    return()



def convertAstropy( ra, dec ) : 
    from astropy.coordinates import SkyCoord
    from astropy import units as u

    if isinstance(ra, float):
        coord = SkyCoord( ra, dec, frame="icrs", unit=[u.deg,u.deg] )
    elif ':' in ra:
        coord = SkyCoord( ra, dec, frame="icrs", unit=[u.hour,u.deg] )
    else:
        coord = SkyCoord( ra, dec, frame="icrs", unit=[u.deg,u.deg] )

    raout  = coord.ra.to_string( unit=u.hour, decimal=False, pad=True, sep=':', precision=2 )
    decout  = coord.dec.to_string( unit=u.degree, decimal=False, pad=True, alwayssign=True, sep=':', precision=1 )
    hmsdms = raout + '   ' + decout
    decimaldeg  = coord.to_string( 'decimal', precision=5 )

    print( hmsdms )
    print( decimaldeg )
    return()



if __name__=='__main__':
    main()
