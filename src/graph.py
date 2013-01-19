import pylab

smooth = 4

def draw(name):
    f = open("res/%s" %name)
    data = [float(d) for d in f.readlines()[0].strip().split(' ')]
    print name,data
    avg_data = [sum(data[d-smooth:d])/float(smooth) for d in range(smooth, len(data))]
    pylab.plot(range(len(avg_data)), avg_data)

if __name__ == "__main__":
    files = ["mario_random", "mario_random_forward", "mario_random_stop_forward"]
    map(draw, files)
    pylab.xlabel("Episode")
    pylab.ylabel("Reward")
    pylab.legend(files)
    pylab.show()
