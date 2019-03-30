'''
Created on Jul 22, 2017

@author: loanvo
'''
from html.parser import HTMLParser
from collections import defaultdict
from matplotlib import pyplot as plt

def plot_state_borders():
    with open("/Users/loanvo/datascience/joelgrus/data/states.txt","r") as stateborderfile:
        stateborderinfo = ''.join(stateborderfile.readlines())
    
    state_borders = defaultdict(dict)
    stateName = None
    class MyHTMLParser(HTMLParser):
        def handle_starttag(self, tag, attrs):
            global stateName
            if tag == "state":
                stateName = attrs[0][1]
                stateColor = attrs[1][1]
                state_borders[stateName]["stateColor"] = stateColor 
                state_borders[stateName]["lat"] = list()
                state_borders[stateName]["lng"] = list()
            if tag == "point":
                state_borders[stateName]["lat"].append(float(attrs[0][1]))
                state_borders[stateName]["lng"].append(float(attrs[1][1]))
        def handle_endtag(self, tag):
            global stateName
            if tag == "state":
                stateName = None
    html_parser = MyHTMLParser()
    html_parser.feed(stateborderinfo)
    """
    print(state_borders["Alabama"]["stateColor"], sep = "\n")
    print(*state_borders["Alabama"]["lat"], sep = "\n")
    """
    state_map_fig = plt.figure()
    for state_name, state_info in state_borders.items():
        lat = state_info["lat"]
        lng = state_info["lng"]
        #plt.plot(lng, lat, color = state_info["stateColor"],linestyle = "-" )
        plt.plot(lng, lat, color = 'k',linestyle = "-" , linewidth = 3)
        #plt.annotate(state_name, xy=((max(lng)+min(lng))/2,(max(lat)+min(lat))/2),
        #             horizontalalignment='center', verticalalignment='center',)
        #plt.title("US states")
        #plt.axis("off")
    return state_map_fig


