#!/usr/bin/python
import datetime
import calendar

MJDzero = datetime.datetime.utcfromtimestamp(0)+datetime.timedelta(seconds=-float(0x007c95674beb4000)/10000000)

mon = lambda month_num:calendar.month_abbr[month_num]

def datetime_to_MJD(d):
    w = d-MJDzero
    return w.days + (w.seconds + w.microseconds/1e6)/86400.

def MJD_to_datetime(m):
    return MJDzero + datetime.timedelta(days=m)

def MJD_to_year(m):
    dayinyear = 365.242199
    t = MJD_to_datetime(m)
    Y= datetime.datetime(t.year, 1, 1)
    f = (t - Y).days/dayinyear
    return t.year + f

def printmjd( mjd ): 
    print('%.3f'%mjd)
    print( datestring( mjd ) )
    print( utstring( mjd ) )

def datestring( mjd ):
    date = MJD_to_datetime( mjd  )
    return('%i.%02i.%02i'%(date.year, date.month, date.day))

def utstring( mjd ) :
    date = MJD_to_datetime( mjd  )
    fracday = round(mjd%1,1)
    return('UT %i %s%04.1f'%(date.year, mon(date.month), date.day+fracday))


if __name__=='__main__':
    import sys
    mjd = float(sys.argv[1])
    printmjd( mjd )
