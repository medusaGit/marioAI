import math
from collections import defaultdict

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

class FixedPolicyAgent(Agent):

    def agent_init(self, taskSpecString):
        random.seed(5)

        self.best_reward = -10
        self.total_steps = 0

        self.trial_steps = 0
        self.trial_number = 0
        self.trial_start = 0
        self.trial_reward = 0
        self.trial_reward_pos = 0
        self.trial_reward_neg = 0

        self.all_scores = [0]
        self.all_actions = []
        
        self.Q = defaultdict(dict)

        self.debug = True
        self.debug = False
        
    def agent_start(self, observation):
        self.all_actions = []
        self.all_scores = [0]
        self.trial_start = time.time()
        self.trial_steps = 0
        self.trial_number = 0
        self.trial_reward = 0
        self.trial_reward_pos = 0
        self.trial_reward_neg = 0
        return self.get_action(observation)
    
    def agent_step(self, reward, observation):
        self.last_observation = observation

        self.trial_steps += 1
        self.total_steps += 1
        self.trial_reward += reward
        if reward > 0:
            self.trial_reward_pos += reward
        else:
            self.trial_reward_neg += reward
        
        return self.get_action(observation, reward)
    
    def agent_end(self, reward):
        self.trial_number += 1
        self.trial_reward += reward
        self.all_scores.append(self.trial_reward)

        if reward > 0:
            self.trial_reward_pos += reward
        else:
            self.trial_reward_neg += reward

        self.print_stats()

        self.propagate_reward(reward)

    
    def agent_cleanup(self):
        hf.write_score("randomAgent", self.all_scores)
    
    def agent_freeze(self):
        print "agent freeze"
        pass
    
    def agent_message(self,inMessage):
        print "agent message:", inMessage
        return None
    

    def get_action(self, observation, reward = 0):
        return self.getRandomAction()

    def getRandomAction(self, mindir=-1, run = 0):
        action = Action(3, 0)
        # direction (left: -1, right: 1, neither: 0)
        action.intArray[0] = random.randint(mindir,1)
        # jumping (yes: 1, no: 0)
        action.intArray[1] = random.randint(0,1)
        # speed button (on: 1, off: 0)
        action.intArray[2] = random.randint(run,1)
        return action
        
    def print_world(self, s = [], sa = [], ok=100):
        global all_scores, q, all_actions, state, state_arr, \
                observation, mario, monsters
        
        observation = self.last_observation
        monsters = hf.get_monsters(observation)
        mario = hf.get_mario(monsters)
        all_cores = self.all_scores
        all_actions = self.all_actions
        if len(sa) > 0: state_arr = sa
        if len(s) > 0: state = s
        q = self.Q

        print "--------------------------------------------------"
        s = hf.getOkolica(observation,ok,ok,ok,ok)
        print "step: %d     reward: %.2f   " % \
                (self.trial_steps, self.trial_reward)
        print "\n".join(["".join(i) for i in s])
        print "x: %2.2f    y: %2.2f    q-len: %d " % \
                (mario.x, mario.y, len(self.Q))
        print ""

    def print_stats(self):
        time_passed = time.time() - self.trial_start
        self.best_reward = max(self.best_reward,self.trial_reward)
        
        self.print_world()

        print "trial number:      %d -" % (self.trial_number)
        print "number of steps:   %d" % (self.trial_steps)
        print "steps per second:  %d" % (self.trial_steps/time_passed)
        print "trial reward pos:  %.2f" % (self.trial_reward_pos)
        print "trial reward neg:  %.2f" % (self.trial_reward_neg)
        print "trial reward:      %.2f" % (self.trial_reward)
        print "best score so far: %.2f" % (self.best_reward)
        print ""
        
       

if __name__=="__main__":        
    AgentLoader.loadAgent(FixedPolicyAgent())


