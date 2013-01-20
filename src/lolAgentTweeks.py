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
        #random.seed(5)

        self.tweek = 4
        self.best_reward = -10
        self.total_steps = 0

        self.trial_steps = 0
        self.trial_number = 0
        self.trial_start = 0
        self.trial_reward = 0
        self.trial_reward_pos = 0
        self.trial_reward_neg = 0

        self.all_scores = []
        self.trial_actions = []
        
        self.Q = defaultdict(lambda: defaultdict(int))

        self.debug = True
        self.debug = False
        
    def agent_start(self, observation):
        self.trial_actions = []
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
        hf.write_score("lolAgentTweeks_%d"%self.tweek, self.all_scores)
    
    def agent_freeze(self):
        print "agent freeze"
        pass
    
    def agent_message(self,inMessage):
        print "agent message:", inMessage
        return None
    
    def propagate_reward(self, reward):
        beta = 0.7
        alpha = 0.3
        gama = 0.1
        Q = self.Q
        l = len(self.trial_actions)
        for i in range(1,l):
            s = self.trial_actions[-i-1][0]
            a = self.trial_actions[-i-1][1]
            sn = self.trial_actions[-i][0]
            an = self.trial_actions[-i][1]
            
            #Q[s][a] = (1-alpha)*Q[s][a] + alpha*(reward + gama*Q[sn][an])
            #reward *= beta
            
            Q[s][a] += reward/(1+ i/10)


    def get_q_action(self, state):
        action = None
        explore = 0.1
        if state in self.Q and explore/10 < random.random():
            # actions = [ (actionT, score), ...]
            items = self.Q[state].items()
            random.shuffle(items)
            actions = sorted(items, key=lambda x:-x[1])
            ind = 0 # int(random.random()**4 * len(actions))
            if actions[ind][1] > 0:
                action = self.createAction(*actions[ind][0])
        if action == None :
            if len(self.Q) > 10 and explore < random.random():
                a = sorted([(k,j) for i in self.Q.values() 
                        for j,k in i.items()])[-10:]
                r = random.choice(a)
                action = self.createAction(*r[1])
            else: 
                action = self.getRandomAction()
                self.Q[state][tuple(action.intArray)] = 0
        self.trial_actions.append((state,tuple(action.intArray)))
        return action

    

    def get_action(self, observation, reward = 0):

        monsters = hf.get_monsters(observation)
        mario = hf.get_mario(monsters)

        state_arr = hf.getOkolica(observation,4,4,4,4)
        state = state_arr.tostring()
        if (state.find("V") != -1 
                and abs(state.find("V") - state.find("M")) <5):
            f = min(state.find("V"), state.find("M"))
            t = max(state.find("V"), state.find("M")) +1
            state = state[f:t]


        action = self.get_q_action(state)
        self.propagate_reward(reward)

        if self.debug: self.print_world(state, state_arr)
        return action        

    def createAction(self,i,j,k):
        action = Action(3, 0)
        action.intArray[0] = i
        action.intArray[1] = j
        action.intArray[2] = k
        return action

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
        global all_scores, q, trial_actions, state, state_arr, \
                observation, mario, monsters
        
        observation = self.last_observation
        monsters = hf.get_monsters(observation)
        mario = hf.get_mario(monsters)
        all_cores = self.all_scores
        trial_actions = self.trial_actions
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

