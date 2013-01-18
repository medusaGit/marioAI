
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


