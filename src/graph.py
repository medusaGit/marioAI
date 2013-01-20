import pylab
import numpy
from random import randint
import os

fileset = 200
smooth = 20

def draw(name):
    global data
    data = []
    for i in range(100):
        fn = "res/%s_%d_%d" % (name, fileset, i)
        if not os.path.isfile(fn):
            continue
        line = open(fn).readline().strip()
        data.append( map(float,line.split(' ')) )
    
    if len(data) > 0 :
        a = sum(data[0][:smooth])/smooth
        l = len(data)
        for i in range(l,smooth):
            for j in range(l):
                data.append([a]*i+data[j][:-i])
            if len(data) > smooth: 
                break


        avg_data = map(numpy.mean, zip(*data))
        pylab.plot(range(len(avg_data)), avg_data)

if __name__ == "__main__":
    files = ["lolAgent", 
            "lolAgentTweeks_1", 
            "lolAgentTweeks_4", 
            "randomAgent", 
            "randomForwardAgent"]
    map(draw, files)
    pylab.xlabel("Episode")
    pylab.ylabel("Reward")
    legend = ["simple Q learning",
            "inc Q learnering 1",
            "inc Q learnering 4",
            "random",
            "random forward"]
    pylab.legend(legend)
    pylab.show()
