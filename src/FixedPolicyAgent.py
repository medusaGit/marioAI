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

class FixedPolicyAgent(Agent):

    def agent_init(self, taskSpecString):
        random.seed(0)

        self.total_steps = 0
        self.step_number = 0

        self.trial_number = 0
        self.trial_start = 0
        self.trial_reward = 0

        self.all_scores = []
        self.all_actions = []
        
        self.Q = {}

        self.alpha = 0.1
        self.gama = 0.9
        
        self.debug = True
        
        
    def print_world(self, observation):
        global scores, actions, obs, q
        q = self.Q
        scores = self.all_scores
        actions = self.all_actions
        obs = observation 
        if self.debug:
            print "step: %d     reward: %.2f   " % \
                    (self.step_number, self.trial_reward)
            print "".join(observation.charArray)

    def agent_start(self, observation):
        self.all_actions = []
        self.trial_start = time.time()
        self.step_number = 0
        self.trial_reward = 0
        self.print_world(observation)
        return self.get_action(observation)
    
    def agent_step(self, reward, observation):

        self.step_number += 1
        self.total_steps += 1
        self.trial_reward += reward
        
        self.print_world(observation)
        
        self.propagate_reward(reward)
        
        return self.get_action(observation)
    
    def agent_end(self, reward):
        self.trial_reward += reward
        self.all_scores.append(self.trial_reward)

        # compute some statistics about the current trial
        time_passed = time.time() - self.trial_start
        self.trial_number += 1
        
        print "trial number:      %d -" % (self.trial_number)
        print "number of steps:   %d" % (self.step_number)
        print "steps per second:  %d" % (self.step_number/time_passed)
        print "trial reward:      %.2f" % (self.trial_reward)
        print "best score so far: %.2f" % (self.best_trial["score"])
        print ""
        
        #if self.trial_reward > self.best_trial["score"] :
        #    self.Q = copy.deepcopy(self.best_trial["Q"])
        #else:
        #    self.best_trial["score"] = self.trial_reward
        #    self.bset_trial["Q"] = copy.deepcopy(self.Q)

    
    def agent_cleanup(self):
        hf.write_score(self.get_action.func_name, self.all_scores)
        pickle.dump(self.Q, open("q.pkl","w"))
        pass
    
    def agent_freeze(self):
        pass
    
    def agent_message(self,inMessage):
        return None
    
    def propagate_reward(self, reward):
        l = len(self.all_actions)
        for i in range(l):

        while abs(reward) > 0.01 and i >= 0:
            # update reward for the given state
            state = self.all_actions[i][0]
            actionTupple = self.all_actions[i][1]
            curRew = self.Q[state][actionTupple] 
            self.Q[state][actionTupple] += alpha*reward #(1-alpha)*curRew
            reward *= self.gama

    def mario_random(self, observation):
        return self.getRandomAction()
    
    def mario_random_forward(self, observation):
        return self.getRandomAction(1)
    
    def mario_random_stop_forward(self, observation):
        return self.getRandomAction(0)

    def get_q_action(self, state):
        if self.debug: print "q len", len(self.Q)
            
        if state not in self.Q:
            actionScores = {}
            for i in range(-1,2):
                for j in range(0,2):
                    for k in range(1,2):

                        actionScores[(i,j,k)] = 0
            self.Q[state] = actionScores 

        # actions = [ (actionTupple, score), ...]
        items = self.Q[state].items()
        random.shuffle(items)
        actions = sorted(items, key=lambda x:-x[1])
        ind = int(random.random()**3 * len(actions))
        actionTuple = actions[ind][0]
        self.all_actions.append((state,actionTuple))
        
        action = self.createAction(*actionTuple)
        a = action
        return action

    

    def get_action(self, observation):
        """Choose an action according to the fixed policy outlined in the
        docstring of this class.
        """

        monsters = hf.get_monsters(observation)
        mario = hf.get_mario(monsters)

        state_arr = hf.getOkolica(observation.charArray,3,3,3,3)

        # fill monsters
        state = state_arr.tostring()
        for m in monsters:
            if m.m_type in [0, 10, 11]:
                # m is Mario
                continue
            #print m.y,m.x 
            dx = m.x - mario.x
            dy = m.y - mario.y
            if abs(dx) < 4 and abs(dy) < 4:
                state.replace("M","X")
            

        for ground in list("1234567"):
            state = state.replace(ground,"7")

        if state == '':
            self.print_world(observation)
            exit(1)
                
        if self.debug: print "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        if self.debug: print " x: %2d    y: %2d    '%30s'" % \
                (mario.x, mario.y, state)
        action = self.get_q_action(state)

        self.set_globals(observation, state, state_arr)
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
