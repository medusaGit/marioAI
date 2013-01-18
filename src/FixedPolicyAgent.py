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

        # set random seed
        random.seed(0)

        # number of steps since the beginning of this run
        self.total_steps = 0
        # number of steps since the beginning of this trial
        self.step_number = 0
        # time when the current trial began
        self.trial_start = 0
        # sequence of actions taken during the last trial
        self.last_actions = []
        # sequence of actions taken so far during the current trial
        self.cur_actions = []
        
        # when this is True, Mario is pausing for some number of steps
        self.walk_hesitating = False
        
    def agent_start(self, observation):
        self.trial_start = time.time()
        self.step_number = 0
        print "State of the world at time step {}:".format(self.step_number)
        print "".join(observation.charArray)
        return self.get_action(observation)
    
    def agent_step(self, reward, observation):
        self.step_number += 1
        self.total_steps += 1
        print "State of the world at time step {}:".format(self.step_number)
        print "".join(observation.charArray)
        return self.get_action(observation)
    
    def agent_end(self, reward):
        # if there were at least 7 actions before the end of the trial,
        # store all actions besides the last 7 actions
        if len(self.cur_actions) > 7:
            self.last_actions = self.cur_actions[:-7]
        # compute some statistics about the current trial
        time_passed = time.time() - self.trial_start
        print "Trial ended after {} steps.".format(self.step_number)
        print "On average, {} steps per second were executed.".\
            format(self.step_number/time_passed)
    
    def agent_cleanup(self):
        pass
    
    def agent_freeze(self):
        pass
    
    def agent_message(self,inMessage):
        return None
    
    def get_action(self, observation):
        """Choose an action according to the fixed policy outlined in the
        docstring of this class.
        """
        # check if we still follow the actions from the previous trial
        if len(self.last_actions) > self.step_number:
            self.cur_actions.append(self.last_actions[self.step_number])
            return self.cur_actions[-1]
        
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
        # add the action to the trajectory being recorded, so it can be reused
        # in the next trial
        self.cur_actions.append(action)

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
