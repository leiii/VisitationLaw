""" Functions for running the PEPR model defined in the 
    --Univeral visitation law of human mobility-- paper 
    
    (https://www.nature.com/articles/s41586-021-03480-9).
    
"""

import random
import time

import itertools as it
import matplotlib.pyplot as plt
import numpy as np

def levy_flight(num_steps: int, alpha: float) -> np.array:
    """ 
    
    Performs a levy flight in 2D starting at the
    origin (0,0).
    
    Args:
        num_steps: number of step in flight
        alpha: shape parameter in jump distribution
        
    Returns:
        x: np.array of x coordinates of trajectory
        y: np.array of y coordinates of trajectory
    
    """
    
    # Set up 
    x_start, y_start = 0,0
    x,y = [x_start], [y_start]
    x_curr, y_curr = x_start, y_start
    
    # Execute trajectory
    for i in range(num_steps-1):
        r = np.random.pareto(alpha)
        theta = np.random.uniform(0,2*np.pi)
        x_jump, y_jump = r*np.cos(theta), r*np.sin(theta)
        x_curr, y_curr = x_curr + x_jump, y_curr + y_jump
        x.append(x_curr)
        y.append(y_curr)
    x = np.array(x)
    y = np.array(y)
    return x,y


def levy_jump(x_curr: int, y_curr: int, alpha: float, box_size: float) -> [int, int]:
    """
    
    Does a levy jump.
    
    Args:
        x_curr: current x coordinate
        y_curr: current y coordiante
        alpha: shape parameter in jump distribution
        box_size: size of grid box in which process is taking place
        
    Returns:
        x_curr: x coordinate after jump
        y_curr: y coordinate after jump
    
    """
    r = np.random.pareto(alpha)
    theta = np.random.uniform(0,2*np.pi)
    x_jump, y_jump = r*np.cos(theta) / box_size, r*np.sin(theta) / box_size
    x_curr, y_curr = int(x_curr + x_jump), int(y_curr + y_jump)
    return x_curr, y_curr


def revisit(visited_places: dict) -> [int, int, dict]:
    """ Chooses a place to revist, where place i is chosen
        with prob \propto S_i, S_i = number of visits to i
        
        Args:
            visited_places[(x,y)] = number visits to place (x,y)
            
        Returns:
            x_curr: new x coordinate
            y_curr: new y coordiante
            visited_places: updated list of visited places
    """
    
    freqs = np.array(list(visited_places.values()))
    freqs = freqs / (1.0*sum(freqs))
    places_indices = range(len(freqs))
    go_to = np.random.choice(places_indices, p=freqs)
    x_curr, y_curr = list(visited_places.keys())[go_to]
    visited_places[(x_curr,y_curr)] += 1
    return x_curr, y_curr, visited_places



def xy_to_cell_id(x: int, y: int, Ngrid: int):
    """ Convert a position (x,y) to the grid cell
        index (where upper left hand column is indexed
        0 & indexing is done rowwise)
    """
    return x + y*Ngrid


def cell_id_to_xy(cell_id: int, Ngrid: int) -> [int, int]:
    """ The reverse of the above function """
    y,x = divmod(cell_id, Ngrid)
    return x,y


def dist(x1: int, y1: int, x2: int, y2: int) -> float:
    """ L2 distance between points
        (x1,y1) and (x2,y2)    
    """
    d = (x2-x1)**2 + (y2-y1)**2
    return np.sqrt(d)


def update_data(x_curr: int, y_curr: int, home: int, data: dict, agent_id, Ngrid: int) -> dict:
    """
    
    The data dictionary contains a tally of 
    all the visitors to a given cell, where
    cells are index from 0, 1, 2, ... N_cells:
    
    data[cell_id] = [agent_id, f, home, r, E]
    
    So data[7] = [ [10, 2, 13, 45, 90], [2, 5, (3,3), 10, 100] ]
        
    Means there were two visitors to cell 7.
    The first visitor was agent 10, visited 
    the cell twice, has home at cell 13
    which is a distance 45 from cell 7, and 
    has expended E = r*f = 45*2 = 90 travel
    energy units in traveling to cell 7.

    Args:
        x_curr: x coordinate of current position of agent
        y_curr: y coordinate of current position of agent
        home: grid cell index of home of current agent
        data: defined above
        agent_id: ID of agent
        
    Returns:
        data: updated data dictionary
        
    """
    
    f = 1  # we know it's a new place
    x_home, y_home = cell_id_to_xy(home, Ngrid)
    r = dist(x_curr, y_curr, x_home, y_home)
    key = xy_to_cell_id(x_curr, y_curr, Ngrid)
    val = [agent_id, f, home, r, r*f]
    
    # If first visit to cell update
    if key not in data:
        data[key] = [val]
        
    # If not, then grab all agent 
    # features vectors at the cell
    # and update the given agent ID's
    # featutre vector; by feature vector
    # I mean [agent_id, f, home, r, E]
    else:
        rows = data[key]
        for i,row in enumerate(rows):
            if row[0] == agent_id:
                [agent_id, f, home, r, E] = row            
                new_row = [agent_id, f+1, home, r, r*(f+1)]
                data[key][i] = new_row
                return data
        data[key].append(val)
    return data


def add_to_visited_places(x_curr: int, y_curr: int, visited_places: dict) -> dict:
    """
    Visited placs[(x,y)] contains the number of visits 
    that the cell (x,y). This updates the count to the
    current position (x_curr, y_curr)
    
    Args:
        x_curr: x coordinate of current position
        y_curr: y coordinate of current position
        visited_place: dict defined above
        
    Returns:
        visited_place: updated dict
    """
    
    if (x_curr, y_curr) not in visited_places:
        visited_places[(x_curr, y_curr)] = 1
    else:
        visited_places[(x_curr, y_curr)] += 1
    return visited_places



def merge(v1: dict, v2: dict) -> dict:
    """ Merges two dictionaries """
    
    for key in v2:
        if key not in v1:
            v1[key] = v2[key]
        else:
            v1[key] += v2[key]
    return v1


def levy_jump_with_PE(x_curr: int, y_curr: int, alpha: float, R: float, nu: float, box_size: int, data: dict, Ngrid: int):
    """ Does a levy flight, except now the 
        angle is chosen according to Preferential 
        Exploration. 
        
        Args:
            x_curr: current x coordinate
            y_curr: current y coordiante
            alpha: shape parameter in jump distribution
            R: sensing radius (see defintion of Preferntial exploration)
            nu: asymmetry parameter (see defintion of Preferntial exploration))
            box_size: size of grid box in which process is taking place
            data: data[cell_id] = [ f_agent1, f_agent2 ] contains list of 
                  feature vectors of agents that have visited that cell
                  where f_agent1 = [agent_id, f - frequency of visit, home cell , r-distance from home cell to cell , r*f]
            Ngrid: number of grid
            
        Returns:
            x_curr: x coordinate after jump
            y_curr: y coordinate after jump
        
    """
    
    r = np.random.pareto(alpha)
    theta = sample_angle(x_curr, y_curr, data, R, nu, Ngrid)
    x_jump, y_jump = r*np.cos(theta) / box_size, r*np.sin(theta) / box_size
    x_curr, y_curr = int(x_curr + x_jump), int(y_curr + y_jump)
    return x_curr, y_curr


def find_neighbours(x_curr: int, y_curr: int, R: int, Ngrid: int) -> [(int, int), (int,int)]:
    """ Return all neighbours on a grid
        in the first R layers
        
        So if R = 1, then you return
        the eight neighbours surrounding
        a given cell 
        
        Auxiliary function for 'sample_angle' 
        method defined below
        
    """
    
    neighbours = [(x_curr + col, y_curr + row) for row in range(-R,R+1) for col in range(-R,R+1) \
              if 0 <= x_curr + col <= Ngrid-1 and 0 <= y_curr + row <= Ngrid-1 ]
    if len(neighbours) > 0:
        neighbours.remove((x_curr, y_curr))
    return neighbours


def get_energies(neighbours: list, data: dict, Ngrid: int) -> list:
    """ 
    Grabs all the energies of the neighbour cells
    Auxilary functions for 'sample_angle' method below 
    """
    Es = np.ones(len(neighbours))
    for i,n in enumerate(neighbours):
        key = xy_to_cell_id(n[0], n[1], Ngrid)
        E = 0
        if key not in data:
            Es[i] = E
        else:
            for row in data[key]:
                E += row[-1]
            Es[i] += E
    return Es


def sample_angle(x_curr: int, y_curr: int, data: dict, R: int, nu: float, Ngrid: int) -> float :
    """
    
    An agent following preferential exploration jumps a distance r
    in a direction theta, where
    
    r ~ Pareto(alpha)   (just like a levy flight)
    theta ~ E(theta;R)^nu     (see paper)
    
    
    where E is the aggregate energy or all 
    cells within a distance R (see paper)
    
    This method samples the angle theta
   
    """
    
    if R == 0:
        return np.random.uniform(0,2*np.pi)

    # Find which neighbour to jump to
    neighbours = find_neighbours(x_curr, y_curr,R, Ngrid)
    energies = get_energies(neighbours, data, Ngrid)
    energies += np.ones(len(energies))
    energies = energies**nu
    
    if sum(energies) == 0:
        index_of_chosen_neighbour = np.random.choice(range(len(neighbours)))
    else:
        energies /= sum(energies)
        index_of_chosen_neighbour = np.random.choice(range(len(neighbours)), p = energies)

    # Convert this to a jump angle
    x1,y1 = x_curr, y_curr
    (x2,y2) = neighbours[index_of_chosen_neighbour]
    angle = find_angle(x1,y1,x2,y2)
    
    # I need to fill in the missing angles here
    # Now I want the final angle to be Uniform(angle-X, angle+X)
    # where X is the nearest angle. 
    angle_to_neighbours = [abs(find_angle(x1,y1,x2,y2) - angle) for (x2,y2) in neighbours if (x2,y2) != (x1,y1)]
    angle_to_neighbours = [x for x in angle_to_neighbours if x != 0]
    X = min(angle_to_neighbours)
    angle_final = np.random.uniform(angle-X,angle+X)
    return angle_final


def find_angle(x1:int, y1: int, x2: int, y2: int) -> float:
    """
    Finds the angle betwen the two points
    (x1, y1) and (x2, y2)
    """
    
    # Find angle
    dx, dy = x2-x1, y2-y1
    r = np.sqrt( dx**2 + dy**2 )
    angle = np.arccos(dx / r)
    
    # Find quandrant
    if dy < 0:
        angle = np.pi + angle
    return angle


def clean_Es(Es):
    """ Auxiliary method """
    return [ x for x in Es if x != 0 ]


def preferential_exploration(num_steps: int, data: dict, alpha: float, rho: float, gamma: float, R: int, nu: float, x_curr: int, y_curr: int, agent_id: int, Ngrid: int, box_size: int) -> dict:
    """ 
    
    Performs preferential exploration for a single agent. See paper for defintion
    of the process.
    
    Args:
        num_steps: number of steps in simulation
        
        data: data[cell_id] = [ f_agent1, f_agent2 ] contains list of 
              feature vectors of agents that have visited that cell
              where the 'feature vector' is
              f_agent1 = [agent_id, f - frequency of visit, home cell , r-distance from home cell to cell , r*f]
        
        R: Sensing radius (see definition of Preferntial exploration)
        
        nu: Model parameter (see definition of Preferential exploration)
        
        x_curr: x coordinate of current position
        
        y_curr: y coordiante of current position
        
        agent_id: ID of agent doing the PERP
       
        Ngrid: as implied
        
        box_size: as implied (size of grid cell)
        
        
    Returns:
        data: updated with trajectory of walker
        
    
    """
    
    # Update the data dictionary 
    home = xy_to_cell_id(x_curr,y_curr,Ngrid)
    f, r, E = 1,0,0
    val = [agent_id, f, home, r, E]
    if home not in data:
        data[home] = [val]
    else:
        data[home].append(val)
        
    # Set up the dict of visited places
    # Need this for the preferential 
    # return part of the mechanism
    visited_places = {}  # {(x,y):freq}
    visited_places[(x_curr,y_curr)] = 1

    # Do walk
    for i in range(num_steps-1):
        
        # Find odds of exploring new location
        num_visited = len(visited_places)
        prob_new = rho*num_visited**(-gamma)   # defined in the Song model
        temp = np.random.rand()

        # Go to new location
        if temp <= prob_new:
            x_curr, y_curr = levy_jump_with_PE(x_curr, y_curr, alpha, R, nu, box_size, data, Ngrid)

            #If jump has taken you outside the box, stop    
            if  x_curr < 0 or x_curr >= Ngrid or y_curr < 0 or y_curr >= Ngrid:
                break
            visited_places = add_to_visited_places(x_curr, y_curr, visited_places)
            data = update_data(x_curr, y_curr, home, data, agent_id, Ngrid) 
            
        # Return to previously visited location   
        else:
            x_curr, y_curr, visited_places = revisit(visited_places)
            cell_id = xy_to_cell_id(x_curr, y_curr, Ngrid)
            list_of_agents = data[cell_id]
        
            # find index of agent
            for j in range(len(list_of_agents)):
                if list_of_agents[j][0] == agent_id:
                    break
                    
            # then update that list
            [agent_id, f, home, r, E] = list_of_agents[j]            
            new_row = [agent_id, f+1, home, r, r*(f+1)]
            data[cell_id][j] = new_row
            
    # walk is done and data has been updated
    # so just return
    return data


def spatial_plot(data: dict, homes: list, Ngrid: int) -> None:
    """ This plots various quantities at each
        cell in an (x,y) grid
    
        1) The total visitation (number of visits to a give) 
        2) The effective travel distance per visitor to that cell
        3) Home locations
        
        
        This method is used after the main simulation
        is run. It plots the data collected.
        
        Args:
            data: defined in above method
            homes: list of homes (as cell indices) of agents
            Ngrid: number
            
        Returns:
            Plots a Figure inline
    
    """
    
    V, E = np.zeros((Ngrid, Ngrid)), np.zeros((Ngrid, Ngrid)),
    for key in data.keys():

        #Find visitation
        vals = data[key]
        x,y = cell_id_to_xy(key,Ngrid)
        visitation = len(vals) 
        V[x][y] = visitation

        #Find energy
        Es = []
        for agent_ID, f_temp, home, r, E1 in vals:
            Es.append(E1)
        E_mean = np.mean(Es)
        E[x][y] = E_mean

    #Homes    
    H = np.zeros((Ngrid, Ngrid))
    for x,y in homes:
        H[x][y] += 1

    plt.figure(figsize=(12,6))
    ax1 = plt.subplot(131)
    plt.imshow(V)
    plt.title('Visitation')

    ax2 = plt.subplot(132)
    plt.imshow(E)
    plt.title('Effective travel distance per visitor')

    ax3 = plt.subplot(133)
    plt.imshow(H)
    plt.title('Homes')
    return
