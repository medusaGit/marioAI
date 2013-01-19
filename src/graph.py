import pylab
import os

smooth = 4
fileset = 10

def draw(name):
    data = []
    for i in range(100):
        fn = "res/%s_%d_%d" % (name, fileset, i)
        if not os.path.isfile(fn):
            break
        line = open(fn).readline().strip()
        data.append( map(float,line.split(' ')) )

    if len(data) > 0 :
        avg_data = map(sum, zip(*data))
        pylab.plot(range(len(avg_data)), avg_data)

if __name__ == "__main__":
    files = ["mario_simple_learner","mario_random", "mario_random_forward", "mario_random_stop_forward"]
    map(draw, files)
    pylab.xlabel("Episode")
    pylab.ylabel("Reward")
    pylab.legend(files)
    pylab.show()
