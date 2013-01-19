import math
import copy
import random
import sys
import time
import numpy as np
import cPickle as pickle

import helperFunctions as hf
reload(hf)

from rlglue.agent.Agent import Agent
from rlglue.types import Action
from rlglue.types import Observation
from rlglue.agent import AgentLoader as AgentLoader
from rlglue.utils.TaskSpecVRLGLUE3 import TaskSpecParser

class LolAgent(Agent):

    def agent_init(self, taskSpecString):
        self.debug = False
        self.reward = 0
        self.stateAction = []
        self.rewards = []
        self.lastState = None
        self.lastAction = []

        print "Starting agent: %s" taskSpecString
        
    def agent_cleanup(self):
        hf.write_score("lol_agent", self.all_scores)
        
    def print_world(self):
        if self.debug:
            print "".join(self.observation.charArray)

    def agent_start(self, observation):
        self.reward = 0
        return self.getAction(observation)

    def addReward(r):
        for i in range(1,1+len(self.lastAction)):
            lastAction[-i] += r/(1+i/10.)
        self.reward += r
    
    def agent_step(self, reward, observation):
        self.addReward(reward)
        return self.getAction(observation)
    
    def agent_end(self, reward):
        self.addReward(reward)
        self.lastAction = []
        self.print_stats()

    def print_stats():
        print "trial number:      %d -" % (self.trial_number)
        print "number of steps:   %d" % (self.step_number)
        print "steps per second:  %d" % (self.step_number/time_passed)
        print "total reward:      %.2f" % (self.total_reward)
        print "trial reward:      %.2f" % (self.trial_reward)
        print "best score so far: %.2f" % (self.best_trial["score"])
        print ""
    
    
    def agent_freeze(self):
        pass
    
    def agent_message(self,inMessage):
        print "Agent message:",inMessage
        return None
    

    def get_action(self, observation):
        
        monsters = hf.get_monsters(observation)
        mario = hf.get_mario(monsters)
        state = hf.getOkolica(observation.charArray)
        return 0

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
    AgentLoader.loadAgent(LolAgent())
