import numpy as np
import matplotlib.pyplot as plt
import itertools as it
import seaborn as sns
import pandas as pd
import random,time

def levy_flight(num_steps,alpha):
    x_start, y_start = 0,0
    x,y = [x_start], [y_start]
    x_curr, y_curr = x_start, y_start
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


def levy_jump(x_curr, y_curr, alpha, box_size):
    r = np.random.pareto(alpha)
    theta = np.random.uniform(0,2*np.pi)
    x_jump, y_jump = r*np.cos(theta) / box_size, r*np.sin(theta) / box_size
    x_curr, y_curr = int(x_curr + x_jump), int(y_curr + y_jump)
    return x_curr, y_curr


def revisit(visited_places):
    """ Chooses a place to revist, where place i is chosen
        with prob \propto S_i, S_i = number of visits to i
    """
    
    freqs = np.array(list(visited_places.values()))
    freqs = freqs / (1.0*sum(freqs))
    places_indices = range(len(freqs))
    go_to = np.random.choice(places_indices, p=freqs)
    x_curr, y_curr = list(visited_places.keys())[go_to]
    visited_places[(x_curr,y_curr)] += 1
    return x_curr, y_curr, visited_places


def preferential_return_old(x_start,y_start,alpha,rho,gamma, Ngrid,num_steps, box_size):
    
    #Parameters    
    x,y = [x_start], [y_start]
    x_curr, y_curr = x_start, y_start
    visited_places = {}  # {(x,y):freq}
    visited_places[(x_start,y_start)] = 1
    
    #Preferential return
    for i in range(num_steps-1):
        
        #Find odds of exploring new location
        num_visited = len(visited_places)
        prob_new = rho*num_visited**(-gamma)   #defined in the Song model
        temp = np.random.rand()
        
        #Go to new location
        if temp <= prob_new:
            x_curr, y_curr = levy_jump(x_curr, y_curr, alpha, box_size)
            
            #If jump has taken you outside the box, stop    
            if  x_curr < 0 or x_curr >= Ngrid or y_curr < 0 or y_curr >= Ngrid:
                return visited_places
                
            #Add to new places    
            if (x_curr, y_curr) not in visited_places:
                visited_places[(x_curr, y_curr)] = 1
            else:
                visited_places[(x_curr, y_curr)] += 1
                
        #Return to previously visted location   
        else:
            x_curr, y_curr, visited_places = revisit(visited_places)  

    return visited_places


def xy_to_cell_id(x,y,Ngrid):
    return x + y*Ngrid


def cell_id_to_xy(cell_id, Ngrid):
    y,x = divmod(cell_id, Ngrid)
    return x,y


def dist(x1,y1,x2,y2):
    d = (x2-x1)**2 + (y2-y1)**2
    return np.sqrt(d)


def find_cell_with_highest_count(data):
    max_so_far = 0 
    for key in data:
        val = data[key]
        if len(val) > max_so_far:
            key_max = key
            max_so_far = len(val)
    return key_max


def update_data(x_curr, y_curr, home, data, agent_id, Ngrid):
    
    f = 1  #we know its a new place
    x_home, y_home = cell_id_to_xy(home, Ngrid)
    r = dist(x_curr, y_curr, x_home, y_home)
    key = xy_to_cell_id(x_curr, y_curr, Ngrid)
    val = [agent_id, f, home, r, r*f]
    if key not in data:
        data[key] = [val]
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


def add_to_visited_places(x_curr, y_curr, visited_places):
    if (x_curr, y_curr) not in visited_places:
        visited_places[(x_curr, y_curr)] = 1
    else:
        visited_places[(x_curr, y_curr)] += 1
    return visited_places


def grab_rs_for_given_f(data,cell,f):
    vals = data[cell]
    rs = []
    for agent_ID, f_temp, home, r, E in vals:
        if f_temp == f:
            rs.append(r)
    return rs


def grab_Es(data,cell):
    vals = data[cell]
    Es = []
    for agent_ID, f_temp, home, r, E in vals:
        Es.append(E)
    return Es


def preferential_return(num_steps,data,alpha,rho,gamma,x_curr,y_curr,agent_id,Ngrid,box_size):
    
    #Update the data
    home = xy_to_cell_id(x_curr,y_curr,Ngrid)
    f, r, E = 1,0,0
    val = [agent_id, f, home, r, E]
    if home not in data:
        data[home] = [val]
    else:
        data[home].append(val)
    visited_places = {}  # {(x,y):freq}
    visited_places[(x_curr,y_curr)] = 1

    for i in range(num_steps-1):
        
        #Find odds of exploring new location
        num_visited = len(visited_places)
        prob_new = rho*num_visited**(-gamma)   #defined in the Song model
        temp = np.random.rand()

        #Go to new location
        if temp <= prob_new:
            x_curr, y_curr = levy_jump(x_curr, y_curr, alpha, box_size)

            #If jump has taken you outside the box, stop    
            if  x_curr < 0 or x_curr >= Ngrid or y_curr < 0 or y_curr >= Ngrid:
                break
            visited_places = add_to_visited_places(x_curr, y_curr, visited_places)
            data = update_data(x_curr, y_curr, home, data, agent_id, Ngrid) 
            
        #Return to previously visited location   
        else:
            x_curr, y_curr, visited_places = revisit(visited_places)
            cell_id = xy_to_cell_id(x_curr, y_curr, Ngrid)
            #print x_curr, y_curr, cell_id
            #print data.keys()
            list_of_agents = data[cell_id]
        
            #find index of 
            for j in range(len(list_of_agents)):
                if list_of_agents[j][0] == agent_id:
                    break
                    
            #then update that list
            [agent_id, f, home, r, E] = list_of_agents[j]            
            new_row = [agent_id, f+1, home, r, r*(f+1)]
            data[cell_id][j] = new_row
            
    return data


def merge(v1, v2):
    """ merges two dictionaries """
    
    for key in v2:
        if key not in v1:
            v1[key] = v2[key]
        else:
            v1[key] += v2[key]
    return v1


def make_freq_matrix(visited_places, Ngrid):
    freqs = np.zeros((Ngrid, Ngrid))
    for key in visited_places:
        freqs[key[0]][key[1]] += visited_places[key]
    return freqs



def levy_jump_with_PE(x_curr, y_curr, alpha, R, nu, box_size, data, Ngrid):
    """ Does a levy flight, except now the 
        angle is chosen according to Preferential 
        Exploration. 
    """
    
    r = np.random.pareto(alpha)
    theta = sample_angle(x_curr, y_curr, data, R, nu, Ngrid)
    x_jump, y_jump = r*np.cos(theta) / box_size, r*np.sin(theta) / box_size
    x_curr, y_curr = int(x_curr + x_jump), int(y_curr + y_jump)
    return x_curr, y_curr


def find_neighbours(x_curr, y_curr, R, Ngrid):
    
    """ Return all neighbours on a grid
        in the first R layers
        
        So if R = 1, then you return
        the eight neighbours surrounding
        a given cell
    """
    
    neighbours = [(x_curr + col, y_curr + row) for row in range(-R,R+1) for col in range(-R,R+1) \
              if 0 <= x_curr + col <= Ngrid-1 and 0 <= y_curr + row <= Ngrid-1 ]
    if len(neighbours) > 0:
        neighbours.remove((x_curr, y_curr))
    return neighbours


def get_energies(neighbours, data, Ngrid):
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


def sample_angle(x_curr, y_curr, data, R, nu, Ngrid):
    
    if R == 0:
        return np.random.uniform(0,2*np.pi)

    #Find which neighbour to jump to
    neighbours = find_neighbours(x_curr, y_curr,R, Ngrid)
    energies = get_energies(neighbours, data, Ngrid)
    energies += np.ones(len(energies))
    energies = energies**nu
    
    if sum(energies) == 0:
        index_of_chosen_neighbour = np.random.choice(range(len(neighbours)))
    else:
        energies /= sum(energies)
        index_of_chosen_neighbour = np.random.choice(range(len(neighbours)), p = energies)

    #Covert this to a jump angle
    x1,y1 = x_curr, y_curr
    (x2,y2) = neighbours[index_of_chosen_neighbour]
    angle = find_angle(x1,y1,x2,y2)
    
    #I need to fill in the missing angles here
    #Now I want the final angle to be Uniform(angle-X, angle+X)
    #where X is the nearest angle. 
    angle_to_neighbours = [abs(find_angle(x1,y1,x2,y2) - angle) for (x2,y2) in neighbours if (x2,y2) != (x1,y1)]
    angle_to_neighbours = [x for x in angle_to_neighbours if x != 0]
    X = min(angle_to_neighbours)
    angle_final = np.random.uniform(angle-X,angle+X)
    return angle_final


def find_angle(x1,y1,x2,y2):
    
    #Find angle
    dx, dy = x2-x1, y2-y1
    r = np.sqrt( dx**2 + dy**2 )
    angle = np.arccos(dx / r)
    
    #Find quandrant
    if dy < 0:
        angle = np.pi + angle
    return angle


def clean_Es(Es):
    return [ x for x in Es if x != 0 ]


def preferential_exploration(num_steps,data,alpha,rho,gamma,R, nu, x_curr,y_curr,agent_id,Ngrid,box_size):
    
    #Update the data
    home = xy_to_cell_id(x_curr,y_curr,Ngrid)
    f, r, E = 1,0,0
    val = [agent_id, f, home, r, E]
    if home not in data:
        data[home] = [val]
    else:
        data[home].append(val)
    visited_places = {}  # {(x,y):freq}
    visited_places[(x_curr,y_curr)] = 1

    for i in range(num_steps-1):
        
        #Find odds of exploring new location
        num_visited = len(visited_places)
        prob_new = rho*num_visited**(-gamma)   #defined in the Song model
        temp = np.random.rand()

        #Go to new location
        if temp <= prob_new:
            x_curr, y_curr = levy_jump_with_PE(x_curr, y_curr, alpha, R, nu, box_size, data, Ngrid)

            #If jump has taken you outside the box, stop    
            if  x_curr < 0 or x_curr >= Ngrid or y_curr < 0 or y_curr >= Ngrid:
                break
            visited_places = add_to_visited_places(x_curr, y_curr, visited_places)
            data = update_data(x_curr, y_curr, home, data, agent_id, Ngrid) 
            
        #Return to previously visited location   
        else:
            x_curr, y_curr, visited_places = revisit(visited_places)
            cell_id = xy_to_cell_id(x_curr, y_curr, Ngrid)
            list_of_agents = data[cell_id]
        
            #find index of 
            for j in range(len(list_of_agents)):
                if list_of_agents[j][0] == agent_id:
                    break
                    
            #then update that list
            [agent_id, f, home, r, E] = list_of_agents[j]            
            new_row = [agent_id, f+1, home, r, r*(f+1)]
            data[cell_id][j] = new_row
    return data


def spatial_plot(data, homes, Ngrid):
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


def run(num_steps,num_trials,box_size,alpha,rho,gamma,R,nu,Ngrid,num_agents):
    
    #Setup
    n_lower, n_upper = int(0.25*Ngrid), int(0.75*Ngrid)
    possible_homes = [(i,j) for i in range(n_lower, n_upper) for j in range(n_lower,n_upper)]
    homes = [random.choice(possible_homes) for _ in range(num_agents)]

    #Do simulation
    data = {}
    t1 = time.time()
    for i,(x_start, y_start) in enumerate(homes):
        agent_id = i
        x_curr, y_curr = x_start, y_start
        data = preferential_exploration(num_steps,data,alpha,rho,gamma,R,nu,x_curr,y_curr,agent_id,Ngrid,box_size)
    t2 = time.time()
    print('took ' + str( (t2-t1)/60.0 ) + ' mins')
    
    return data, homes



def plot_rf(data, par):
    
    alpha, rho, gamma, num_agents = par
    
    #Stuff for plotting
    num_bins = 20
    fs = [1,2,3,4,5]

    plt.figure(figsize=(20,5))
    plt.subplot(131)


    #Plot N versus r
    for f1 in fs:
        rs = []
        for cell in data.keys():
            vals = data[cell]
            for agent_ID, f_temp, home, r, E in vals:
                if f_temp == f1:
                    if r != 0:
                        rs.append(r)
        #Plot rs
        bins = np.linspace(min(rs),10**2,num_bins)
        frq, edges = np.histogram(rs, bins)
        mid = [0.5*(edges[i] + edges[i+1]) for i in range(len(edges)-1) ]
        plt.loglog(mid, frq,'o-')
    plt.legend(['f = ' + str(f1) for f1 in fs])
    plt.xlabel('$r$', fontsize=18)
    plt.ylabel('$N_f(r)$', fontsize=18)


    #Plot N versus r*f
    plt.subplot(132)
    for f1 in fs:
        rs = []
        for cell in data.keys():
            vals = data[cell]
            for agent_ID, f_temp, home, r, E in vals:
                if f_temp == f1:
                    if r != 0:
                        rs.append(r*f_temp)
        #Plot rs
        bins = np.linspace(min(rs),10**2,num_bins)
        frq, edges = np.histogram(rs, bins)
        mid = [0.5*(edges[i] + edges[i+1]) for i in range(len(edges)-1) ]
        plt.loglog(mid, frq,'o-')
    plt.legend(['f = ' + str(f1) for f1 in fs])
    plt.xlabel('$r f$', fontsize=18)


    #Plot N versus r*f^2
    plt.subplot(133)
    for f1 in fs:
        rs = []
        for cell in data.keys():
            vals = data[cell]
            for agent_ID, f_temp, home, r, E in vals:
                if f_temp == f1:
                    if r != 0:
                        rs.append(r*f_temp**2)

        #Plot rs
        bins = np.linspace(min(rs),10**3,num_bins)
        frq, edges = np.histogram(rs, bins)
        mid = [0.5*(edges[i] + edges[i+1]) for i in range(len(edges)-1) ]
        plt.loglog(mid, frq,'o-')
    plt.legend(['f = ' + str(f1) for f1 in fs])
    plt.xlabel('$r f^2$', fontsize=18)
    filename = 'figures/EPE_lattice/alpha_{}_rho_{}_gamma_{}_Nagent_{}.pdf'.format(alpha,rho,gamma,num_agents)
    plt.savefig(filename)