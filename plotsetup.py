from matplotlib import rcParams, rc

def anyfig( ):
    """
    set the rcParams settings suitable for making figures
    """
    rcParams['text.usetex'] = True   # We like TeX
    rcParams['font.family']='serif'  # e.g. Times
    rcParams['savefig.dpi']=300       # higher res outputs



def paperfig(bigtext=False):
    """
    set the rcParams settings suitable for
    paper figures
    """
    anyfig()

    if bigtext :
        rcParams['font.size']=16    # text big enough for a half-page fig
        rcParams['axes.labelsize'] = 16      # fontsize of the x and y labels
        rcParams['axes.titlesize'] = 16      # fontsize of the axes title
        rcParams['xtick.labelsize'] = 14      # fontsize of the x and y labels
        rcParams['ytick.labelsize'] = 14      # fontsize of the x and y labels
    else :
        rcParams['font.size']=12    # text big enough for a half-page fig
        rcParams['axes.labelsize'] = 12      # fontsize of the x and y labels
        rcParams['axes.titlesize'] = 12      # fontsize of the axes title
        rcParams['xtick.labelsize'] = 11      # fontsize of the x and y labels
        rcParams['ytick.labelsize'] = 11      # fontsize of the x and y labels

    rcParams['lines.linewidth']=1.5
    rcParams['lines.color']='k'     # thicker black lines 
    rcParams['lines.markersize']=6
    rcParams['lines.markeredgewidth']=1.     # big fat points
    rcParams['axes.linewidth']=1.
    rcParams['xtick.major.width'] = 1.5      # major tick thickness in points
    rcParams['xtick.minor.width'] = 1.2      # major tick thickness in points
    rcParams['xtick.major.size'] = 6      # major tick size in points
    rcParams['xtick.minor.size'] = 4      # minor tick size in points
    rcParams['xtick.major.pad'] = 8      # distance to major tick label in points
    rcParams['xtick.minor.pad'] = 8      # distance to the minor tick label in points

    rcParams['ytick.major.width'] = 1.5      # major tick thickness in points
    rcParams['ytick.minor.width'] = 1.2      # major tick thickness in points
    rcParams['ytick.major.size'] = 6      # major tick size in points
    rcParams['ytick.minor.size'] = 4      # minor tick size in points
    rcParams['ytick.major.pad'] = 8      # distance to major tick label in points
    rcParams['ytick.minor.pad'] = 8      # distance to the minor tick label in points


def halfpaperfig(newfig=None, figsize=[]):
    """
    set the rcParams settings suitable for
    paper figures
    """
    paperfig()

    rcParams['figure.figsize']=[4.5,4.5]

    if len(figsize) > 0 :
        rcParams['figure.figsize']=figsize

    rcParams['figure.subplot.wspace']=0.2
    rcParams['figure.subplot.hspace']=0.2
    rcParams['figure.subplot.left']=0.15
    rcParams['figure.subplot.bottom']=0.1
    rcParams['figure.subplot.top']=0.95
    rcParams['figure.subplot.right']=0.95

    if newfig: 
        from pylab import figure, close
        close(newfig)
        fig = figure(newfig)
        return( fig )
    else : 
        from pylab import gcf
        return( gcf() )



def posterfig(newfig=None):
    """
    set the rcParams settings suitable for
    poster figures
    """
    from pylab import rcParams

    rcParams['figure.figsize']=(4,3)

    rcParams['text.usetex']=False     # We like TeX, but TeX doesn't work on sn
#    rc('text', dvipnghack=True)    # TeX fix?
    rcParams['font.weight']='heavy'    # bold fonts are easier to see 
    rcParams['font.size']=20    #  big text
    rcParams['lines.linewidth']=5
    rcParams['lines.color']='k'     # thicker black lines 
    rcParams['lines.markersize']=15
    rcParams['lines.markeredgewidth']=3     # big fat points
#    rc('grid', c='0.5', ls='-', lw=0.5)  # solid gray grid lines
    rcParams['savefig.dpi']=180       # higher res outputs
    rcParams['axes.linewidth']=4
    rcParams['xtick.labelsize']=24
    rcParams['ytick.labelsize']=24     # tick labels bigger
    rcParams['xtick.major.size'] = 6      # major tick size in points
    rcParams['xtick.minor.size'] = 4      # minor tick size in points
    rcParams['xtick.major.pad'] = 8      # distance to major tick label in points
    rcParams['xtick.minor.pad'] = 8      # distance to the minor tick label in points
    rcParams['ytick.major.size'] = 6      # major tick size in points
    rcParams['ytick.minor.size'] = 4      # minor tick size in points
    rcParams['ytick.major.pad'] = 8      # distance to major tick label in points
    rcParams['ytick.minor.pad'] = 8      # distance to the minor tick label in points
    rcParams['axes.labelsize'] = 24      # fontsize of the x and y labels
    rcParams['axes.titlesize'] = 28      # fontsize of the axes title

    if newfig: 
        from pylab import figure, close
        close(newfig)
        figure(newfig)


def presfig(figsize=[], wide=True):
    """ 
    set up for preparing plots for a screen presentation 
    """
    from pylab import rcParams, rc, gcf

    if wide :
        rcParams['figure.figsize']=[9.8,5.5]
        rcParams['figure.subplot.wspace']=0.2
        rcParams['figure.subplot.hspace']=0.2
        rcParams['figure.subplot.left']=0.15
        rcParams['figure.subplot.bottom']=0.2
        rcParams['figure.subplot.top']=0.95
        rcParams['figure.subplot.right']=0.95
    else :
        rcParams['figure.figsize']=[5.5,5.5]
        rcParams['figure.subplot.wspace']=0.2
        rcParams['figure.subplot.hspace']=0.2
        rcParams['figure.subplot.left']=0.2
        rcParams['figure.subplot.bottom']=0.15
        rcParams['figure.subplot.top']=0.95
        rcParams['figure.subplot.right']=0.95

    if len(figsize) > 0 :
        rcParams['figure.figsize']=figsize

    rcParams['savefig.dpi']=100       # lower res outputs
    rcParams['text.usetex']=True     # We like TeX, 
    rcParams['font.weight']='heavy'    # bold fonts are easier to see 
    rcParams['font.family']='serif'    # clear serify text
    rcParams['font.serif']='Palatino,Times'    # clear serify text

    rcParams['font.size']=20    #  big text
    rcParams['lines.linewidth']=3
    rcParams['lines.color']='k'     # thicker black lines 
    rcParams['lines.markersize']=10
    rcParams['lines.markeredgewidth']=2     # big fat points
#    rc('grid', c='0.5', ls='-', lw=0.5)  # solid gray grid lines
    rcParams['axes.linewidth']=3
    rcParams['xtick.labelsize']=30
    rcParams['ytick.labelsize']=30     # tick labels bigger
    rcParams['xtick.major.size'] = 20      # major tick size in points
    rcParams['xtick.minor.size'] = 10      # minor tick size in points
    rcParams['xtick.major.width'] = 3      # major tick size in points
    rcParams['xtick.minor.width'] = 2      # major tick size in points
    rcParams['xtick.major.pad'] = 6    # distance to major tick label in points
    rcParams['xtick.minor.pad'] = 6    # distance to the minor tick label in points
    rcParams['ytick.major.size'] = 20      # major tick size in points
    rcParams['ytick.minor.size'] = 10      # minor tick size in points
    rcParams['ytick.major.width'] = 3      # major tick size in points
    rcParams['ytick.minor.width'] = 2      # major tick size in points
    rcParams['ytick.major.pad'] = 6    # distance to major tick label in points
    rcParams['ytick.minor.pad'] = 6    # distance to the minor tick label in points
    rcParams['axes.labelsize'] = 32      # fontsize of the x and y labels
    rcParams['axes.titlesize'] = 36      # fontsize of the axes title
    fig = gcf()
    return( fig )


def fullpaperfig( figsize=[], **kwargs):
    """
    set the rcParams settings suitable for
    a full-width paper figure
    """
    from pylab import rcParams, gcf

    paperfig( **kwargs )
    rcParams['figure.figsize']=[8,3.5]

    if len(figsize) > 0 :
        rcParams['figure.figsize']=figsize
    fig = gcf()
    return( fig )


def thirdpaperfig( figsize=[]):
    """
    set the rcParams settings suitable for
    a full-width paper figure
    """
    from pylab import rcParams, gcf

    paperfig()
    rcParams['figure.figsize']=[3.5,3.5]

    if len(figsize) > 0 :
        rcParams['figure.figsize']=figsize
    fig = gcf()
    return( fig )


def webfig(newfig=None, figsize=[]):
    """
    set the rcParams settings suitable for
    paper figures
    """
    from pylab import rcParams, rc
    rcParams['figure.figsize']=[8,6]
    if len(figsize) > 0 :
        rcParams['figure.figsize']=figsize
    rcParams['font.size']=13    # text big enough for a half-page fig
    rcParams['font.weight']='bold'    # text big enough for a half-page fig
    rcParams['lines.linewidth']=1.5
    rcParams['lines.color']='k'     # thicker black lines 
    rcParams['lines.markersize']=6
    rcParams['lines.markeredgewidth']=1.     # big fat points
    rcParams['savefig.dpi']=72       # low res outputs
    rcParams['axes.linewidth']=1.
#    rc('xtick', labelsize=15)
#    rc('ytick', labelsize=15)     # tick labels bigger
    rcParams['xtick.major.size'] = 6      # major tick size in points
    rcParams['xtick.minor.size'] = 4      # minor tick size in points
    rcParams['xtick.major.pad'] = 8      # distance to major tick label in points
    rcParams['xtick.minor.pad'] = 8      # distance to the minor tick label in points
    rcParams['xtick.labelsize'] = 13      # fontsize of the x and y labels

    rcParams['ytick.major.size'] = 6      # major tick size in points
    rcParams['ytick.minor.size'] = 4      # minor tick size in points
    rcParams['ytick.major.pad'] = 8      # distance to major tick label in points
    rcParams['ytick.minor.pad'] = 8      # distance to the minor tick label in points
    rcParams['ytick.labelsize'] = 13      # fontsize of the x and y labels

    rcParams['axes.labelsize'] = 13      # fontsize of the x and y labels
    rcParams['axes.titlesize'] = 14      # fontsize of the axes title

    if newfig: 
        from pylab import figure, close
        close(newfig)
        figure(newfig)



def usetex(on=True):
    from pylab import rcParams, rc
    if on : 
        # rc('font',**{'family':'serif','serif':['Palatino','Times']})
        rcParams['text.usetex']=True     # We like TeX
        # rcParams['text.dvipnghack']=False    # TeX fix?
    else : 
        rcParams['text.usetex']=False     # We don't like TeX

def default():
    from pylab import rcdefaults
    rcdefaults()

