import math
import random
import sys
import time

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
    
    def agent_init(self, taskSpecString):
        if taskSpecString.find("Mario-v1") != -1:
            print "Task specification contains Mario-v1"
        else:
            print "Task specification does not contain string Mario-v1"
            exit()

        # possible learners:
        # mario_random
        # mario_random_forward
        # mario_random_stop_forward
        self.learner = self.mario_random_stop_forward

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
        
        self.print_states = False
        
        
    def print_world(self, observation):
        if self.print_states:
            print "State of the world at time step %d:" % (self.step_number)
            print "".join(observation.charArray)


    def agent_start(self, observation):
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
        pass
    
    def agent_freeze(self):
        pass
    
    def agent_message(self,inMessage):
        return None

    def mario_random(self, observation):
        return self.getRandomAction()
    
    def mario_random_forward(self, observation):
        return self.getRandomAction(1)
    
    def mario_random_stop_forward(self, observation):
        return self.getRandomAction(0)

    def mario_q_matrix(self, observation):
        return self.getRandomAction(0)

    def get_action(self, observation):
        """Choose an action according to the fixed policy outlined in the
        docstring of this class.
        """
        
        monsters = hf.get_monsters(observation)
        mario = hf.get_mario(monsters)

        # check the blocks in the area to Mario's upper right
        dollarAbove = False
        for up in range(5):
            for right in range(7):
                tile = hf.get_tile_at(mario.x + right, mario.y + up, observation)
                dollarAbove |= tile == '$'
                if tile in [' ', 'M', None]:
                    # don't worry if it is a blank space
                    pass

        # look for nearby monsters by checking its positions againts Mario's
        for m in monsters:
            if m.m_type in [0, 10, 11]:
                # m is Mario
                continue
            dx = m.x - mario.x
            dy = m.y - mario.y


        action = self.getRandomAction()

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
    AgentLoader.loadAgent(FixedPolicyAgent())
