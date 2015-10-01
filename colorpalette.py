# -*- coding: iso-8859-1 -*-

"""
Colour-blind proof distinct colours module, based on work by Paul Tol
Pieter van der Meer, 2011
SRON - Netherlands Institute for Space Research
"""

black='k'
white='w'
grey='0.5'
lightgrey='0.7'
darkgrey='0.3'

# colour table in HTML hex format
blue='#332288'
darkblue='#332288'

lightblue='#88CCEE'
pearlaqua='#88CCEE'

teal='#44AA99'
bluegreen='#44AA99'
hanblue='#44AA99'

green='#117733'
darkgreen='#117733'
ultramarine='#117733'

olive='#999933'
olivegreen='#999933'
medgreen='#999933'
mountbatten='#999933'

beige='#DDCC77'
chestnut='#DDCC77'
palechestnut='#DDCC77'

pink='#CC6677'
coral='#CC6677'
fuzzywuzzy='#CC6677'
lightred='#CC6677'

maroon='#882255'
red='#882255'
sienna='#882255'

purple='#AA4499'

tyrianred='#661100'
tyrian='#661100'
brickred='#661100'
darkred='#661100'


bluegray='#6699CC'
bluegrey='#6699CC'
cadetblue='#6699CC'

rosequartz='#AA4466'
rose='#AA4466'
medred='#AA4466'

medblue='#4477AA'
darkbluegrey='#4477AA'
darkbluegray='#4477AA'

gold='#DFA53A'
darkyellow='#DFA53A'
darkgold='#DFA53A'
darkgoldenrod='#DFA53A'
yellow='#DFA53A'

brown='#CC6600'

darkorange='darkorange'

hexcols = ['#332288', '#88CCEE', '#44AA99', '#117733', '#999933', '#DDCC77', 
           '#CC6677', '#882255', '#AA4499', '#661100', '#6699CC', '#AA4466',
           '#4477AA']

greysafecols = ['#809BC8', '#FF6666', '#FFCC66', '#64C204']

xarr = [[12], 
        [12, 6], 
        [12, 6, 5], 
        [12, 6, 5, 3], 
        [0, 1, 3, 5, 6], 
        [0, 1, 3, 5, 6, 8], 
        [0, 1, 2, 3, 5, 6, 8], 
        [0, 1, 2, 3, 4, 5, 6, 8], 
        [0, 1, 2, 3, 4, 5, 6, 7, 8], 
        [0, 1, 2, 3, 4, 5, 9, 6, 7, 8], 
        [0, 10, 1, 2, 3, 4, 5, 9, 6, 7, 8], 
        [0, 10, 1, 2, 3, 4, 5, 9, 6, 11, 7, 8]]

def get_distinct(nr):
    """get specified nr of distinct colours in HTML hex format.
    in: nr - number of colours [1..12]
    returns: list of distinct colours in HTML hex
    """
    #
    # check if nr is in correct range
    #
    
    if nr < 1 or nr > 12:
        print "wrong nr of distinct colours!"
        return

    #
    # get list of indices
    #
    
    lst = xarr[nr-1]
    
    #
    # generate colour list by stepping through indices and looking them up
    # in the colour table
    #

    i_col = 0
    col = [0] * nr
    for idx in lst:
        col[i_col] = hexcols[idx]
        i_col+=1
    return col

