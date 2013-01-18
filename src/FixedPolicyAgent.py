#
# FixedPolicyAgent.py
# Contains an implementation of a fixed-policy agent with some randomness and
# without learning for the GeneralizedMario domain.
# 
# Copyright (C) 2009 John Asmuth and Rutgers University
#    - original Java implementation
# Copyright (C) 2012 Tadej Janez
#    - port to Python with some enhancements and corrections
#
# All rights reserved.
#

# uncomment this to use Winpdb remote debugger
#import rpdb2; rpdb2.start_embedded_debugger("agent")

import math, random, sys, time

from rlglue.agent.Agent import Agent
from rlglue.types import Action
from rlglue.types import Observation
from rlglue.agent import AgentLoader as AgentLoader
from rlglue.utils.TaskSpecVRLGLUE3 import TaskSpecParser

def get_tile_at(x, y, observation):
    """Return the char representing the tile at the given location.
    If unknown, return None.

    Valid tiles:
    M - the tile Mario is currently on. There is no tile for a monster.
    $ - a coin
    b - a smashable brick
    ? - a question block
    | - a pipe. Gets its own tile because often there are pirahna plants in them
    ! - the finish line
    an integer in range [0,7] - 3 bit binary mask, where:
        - the first bit means "cannot go through this tile from above"
        - the second bit means "cannot go through this tile from below"
        - the third bit means "cannot go through this tile from either side"

    """
    x = int(x)
    if x < 0:
        return '7'
    y = 16 - int(y)
    x -= observation.intArray[0]
    if x < 0 or x > 21 or y < 0 or y > 15:
        return None
    index = y*22 + x
    return observation.charArray[index]

class Monster():

    """Contains information about a monster."""

    TYPES_TO_NAMES = ["Mario", "Red Koopa", "Green Koopa", "Goomba", "Spikey",
                      "Spikey", "Piranha Plant", "Mushroom", "Fire Flower",
                      "Fireball", "Shell", "Big Mario", "Fiery Mario"]

    def __init__(self, x, y, sx, sy, m_type, winged):
        # x and y positions of the monster
        self.x = x
        self.y = y
        # speed in x and y directions (i. e. instantaneous changes in x and
        # y per step)
        self.sx = sx
        self.sy = sy
        # monster type (0 to 11)
        self.m_type = m_type
        # human recognizable name for the monster
        self.m_name = Monster.TYPES_TO_NAMES[m_type]
        # winged monsters bounce up and down
        self.winged = winged

def get_monsters(observation):
    """Get all monsters from the observation (including Mario).
    Return a list of Monster objects.

    """
    monsters = []
    for i in range(len(observation.intArray[1:])/2):
        m_type = observation.intArray[1 + 2*i]
        winged = observation.intArray[2 + 2*i]
        x, y, sx, sy = observation.doubleArray[4*i : 4*(i+1)]
        monsters.append(Monster(x, y, sx, sy, m_type, winged))
    return monsters

def get_mario(monsters):
    """Get Mario from the list of monsters.
    Return a Monster object or None if Mario is not in the list.

    """
    for monster in monsters:
        if monster.m_type in [0, 10, 11]:
            return monster
    else:
        return None

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
        
        monsters = get_monsters(observation)
        mario = get_mario(monsters)

        # sometimes jump for no reason at all. at the end of this function,
        # the value of this variable will be compared against a random number
        # to see if Mario should jump
        jump_hesitation = 0.95

        # check the blocks in the area to Mario's upper right
        for up in range(5):
            for right in range(7):
                tile = get_tile_at(mario.x + right, mario.y + up, observation)
                if tile == '$':
                    # there is a coin, so jump more often
                    jump_hesitation *= 0.7
                elif tile in [' ', 'M', None]:
                    # don't worry if it is a blank space
                    pass
                else:
                    # tend to jump more if there is a block close
                    jump_hesitation *= 1.0 * right / 7

        # check for a pit (concealed hole in the ground) in front of Mario
        # TODO: Improve this code!
        is_pit = False
        for right in range(3):
            pit_col = True
            for down in range(int(math.ceil(mario.y))):
                tile = get_tile_at(mario.x + right, mario.y - down, observation)
                if tile not in [' ', 'M', None]:
                    pit_col = False
                    break
            if pit_col:
                is_pit = True
                break
        if is_pit:
            # always jump if there is a pit
            jump_hesitation = 0

        # look for nearby monsters by checking its positions againts Mario's
        monster_near = False
        for m in monsters:
            if m.m_type in [0, 10, 11]:
                # m is Mario
                continue
            dx = m.x - mario.x
            dy = m.y - mario.y
            if dx > -1 and dx < 10 and dy > -4 and dy < 4:
                # the more monsters and the closer they are, the more likely
                # Mario is to jump
                jump_hesitation *= (dx + 2)/12
                monster_near = True
        
        # hold down the jump button while in the air sometimes, to jump higher
        if mario.sy > 0.1:
            jump_hesitation *= 0.5

        # check if Mario is already hesitating walking
        if self.walk_hesitating:
            # stop hesitating if there is no monster near (or sometimes by
            # chance)
            if not monster_near or random.random() > 0.75:
                self.walk_hesitating = False
        # sometimes hesitate if there is a monster near
        elif monster_near and random.random() > 0.8:
            self.walk_hesitating = True
        # sometimes hesitate even if there isn't one
        elif random.random() > 0.9:
            self.walk_hesitating = True

        action = Action(3, 0)
        # direction (left: -1, right: 1, neither: 0)
        action.intArray[0] = 0 if self.walk_hesitating else 1
        # jumping (yes: 1, no: 0)
        action.intArray[1] = 1 if random.random() > jump_hesitation else 0
        # speed button (on: 1, off: 0)
        action.intArray[2] = 1 if not (is_pit or monster_near) else 0
        
        # add the action to the trajectory being recorded, so it can be reused
        # in the next trial
        self.cur_actions.append(action)

        return action        

if __name__=="__main__":        
    AgentLoader.loadAgent(FixedPolicyAgent())
