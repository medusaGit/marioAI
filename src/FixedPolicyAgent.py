import math
import random
import sys
import time
import numpy as np
import cPickle as pickle

import helperFunctions as hf

from rlglue.agent.Agent import Agent
from rlglue.types import Action
from rlglue.types import Observation
from rlglue.agent import AgentLoader as AgentLoader
from rlglue.utils.TaskSpecVRLGLUE3 import TaskSpecParser

class FixedPolicyAgent(Agent):

    """A simple agent that:
        - never goes to the left
        - mostly goes to the right
        - tends to jump when under coins
        - jumps if it cannot walk to the right due to a block
        - tends to jump when there is a monster nearby
        - tends to jump when there is a pit nearby
        - tends to run when there is nothing nearby
        Also, it will remember the last trial, and repeat it exactly except
        for the last 7 steps (assuming there are at least 7 steps).

    """

    def __init__(self,learner=""):
        LEARNERS = {
                "mario_random" : \
                        self.mario_random,
                "mario_random_forward" : \
                        self.mario_random_forward,
                "mario_random_stop_forward" : \
                        self.mario_random_stop_forward,
                "mario_simple_learner" : \
                        self.mario_simple_learner 
                }
        
        self.learner = self.mario_simple_learner
        if learner in LEARNERS:
            self.learner = LEARNERS[learner]
        
        

    def agent_init(self, taskSpecString):
        if taskSpecString.find("Mario-v1") != -1:
            print "Task specification contains Mario-v1"
        else:
            print "Task specification does not contain string Mario-v1"
            exit()

        # set random seed
        # random.seed(0)

        # number of all trials
        self.trial_number = 0
        # number of steps since the beginning of this run
        self.total_steps = 0
        # number of steps since the beginning of this trial
        self.step_number = 0
        # time when the current trial began
        self.trial_start = 0
        # total reward
        self.total_reward = 0
        # trial reward
        self.trial_reward = 0

        # scores for all trials
        self.all_scores = []
        
        # matrix for all states and actions
        self.Q = pickle.load(open("q.pkl"))

        # matrix for actions taken in current run
        self.all_actions = []

        # how much current reward afects overall reward
        self.alpha = 0.1

        # rate of reward propagation, smaller means it will affect
        # more states
        self.gama = 0.9
        
        self.debug = False
        
        
    def print_world(self, observation):
        if self.debug:
            print "State of the world at time step %d:" % (self.step_number)
            print "".join(observation.charArray)


    def agent_start(self, observation):
        self.all_actions = []
        self.trial_start = time.time()
        self.step_number = 0
        self.trial_reward = 0
        self.print_world(observation)
        return self.learner(observation)
    
    def agent_step(self, reward, observation):
        if self.debug and reward > 0: 
            print "################################################"
            print reward
            print "################################################"
        self.step_number += 1
        self.total_steps += 1
        self.trial_reward += reward
        self.print_world(observation)
        self.propagate_reward(reward)
        return self.learner(observation)
    
    def agent_end(self, reward):
        self.trial_reward += reward
        self.total_reward += self.trial_reward
        self.all_scores.append(self.trial_reward)

        # compute some statistics about the current trial
        time_passed = time.time() - self.trial_start
        self.trial_number += 1
        
        print "trial number:      %d -" % (self.trial_number)
        print "number of steps:   %d" % (self.step_number)
        print "steps per second:  %d" % (self.step_number/time_passed)
        print "total reward:      %.2f" % (self.total_reward)
        print "trial reward:      %.2f" % (self.trial_reward)
        print ""
    
    def agent_cleanup(self):
        hf.write_score(self.learner.func_name, self.all_scores)
        pickle.dump(self.Q, open("q.pkl","w"))
        pass
    
    def agent_freeze(self):
        pass
    
    def agent_message(self,inMessage):
        return None

    def propagate_reward(self, reward):
        i = len(self.all_actions)-1
        alpha = self.alpha
        while abs(reward) > 0.01 and i >= 0:
            # update reward for the given state
            state = self.all_actions[i][0]
            actionTupple = self.all_actions[i][1]
            curRew = self.Q[state][actionTupple] 
            self.Q[state][actionTupple] = alpha*reward + (1-alpha)*curRew
            reward *= self.gama

    def mario_random(self, observation):
        return self.getRandomAction()
    
    def mario_random_forward(self, observation):
        return self.getRandomAction(1)
    
    def mario_random_stop_forward(self, observation):
        return self.getRandomAction(0)

    def get_q_action(self, state):
        ## there are 10 all possible states to be in
        global q,s,a
        q = self.Q
        s = state
        if self.debug: print "q len", len(q)
            
        if state not in self.Q:
            actionScores = {}
            for i in range(-1,2):
                for j in range(0,2):
                    for k in range(0,2):

                        actionScores[(i,j,k)] = 0
            self.Q[state] = actionScores 

        # actions = [ (actionTupple, score), ...]
        actions = sorted(self.Q[state].items(), key=lambda x:x[1])
        actionTuple = actions[int(random.random()**3 * len(actions))][0]
        self.all_actions.append((state,actionTuple))

        action = self.createAction(*actionTuple)
        a = action
        return action

    def mario_simple_learner(self, observation):
        """Choose an action according to the fixed policy outlined in the
        docstring of this class.
        """

        monsters = hf.get_monsters(observation)
        mario = hf.get_mario(monsters)
        x = np.reshape(np.array(observation.charArray),(16,22))[:,:21]
        state = np.flipud(x)[mario.y-2:mario.y+3 , mario.x-2:mario.x+3]
        state = state.tostring()
        for ground in list("1234567"):
            state = state.replace(ground,"7")

         
        if self.debug: print "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        if self.debug: print mario.x,mario.y,"state", "'%s'" % state
        # print "observeation.charArray", observation.charArray
        action = self.get_q_action(state)
        return action        

    def createAction(self,i,j,k):
        action = Action(3, 0)
        action.intArray[0] = i
        action.intArray[1] = j
        action.intArray[2] = 1
        return action

    def getRandomAction(self, mindir=-1):
        action = Action(3, 0)
        # direction (left: -1, right: 1, neither: 0)
        action.intArray[0] = random.randint(mindir,1)
        # jumping (yes: 1, no: 0)
        action.intArray[1] = random.randint(0,1)
        # speed button (on: 1, off: 0)
        action.intArray[2] = random.randint(0,1)
        return action
        


if __name__=="__main__":        
    lrn = ""
    if len(sys.argv) > 1:
        lrn = sys.argv[1]
    AgentLoader.loadAgent(FixedPolicyAgent(lrn))
