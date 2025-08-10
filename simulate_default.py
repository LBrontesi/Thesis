import numpy as np

def SimulatedDefault(time,parameters,markov_chain,firm_number):
    """
    :param time: vector of time
    :param parameters: parameters of the intensity of default
    :param markov_chain: path of the markov chain, in our case observed
    :param firm_number: number of firms in the simulation
    :return: the counting process, the default indicator of each firm, the vector of state loss and the default intensity value over time
    """
    default_indicator =  np.zeros((len(time),firm_number))
    default_intensities =  np.zeros((len(time),firm_number))
    state_loss = np.zeros((len(time), firm_number))
    for i in range(firm_number):
        default_intensities[0, i] = parameters[i,0,0]
    dt = time[1] - time[0]
    for t in range(1,len(time)):

        for i in range(firm_number):
            h = abs(np.hstack((0,np.diff(np.sum(np.delete(default_indicator,i,1),axis=1)[:t]))))
            time_step = abs(time - time[t])
            if default_indicator[t-1,i]==1 : #the company has defaulted
                default_indicator[t-1:,i]=1
                default_intensities[t,i]= 0
            else: # if the company is still alive, then
                #compute the lambda at time t
                default_intensities[t, i] = np.dot(parameters[i,:, 0], markov_chain[:, t - 1]) + np.dot(parameters[i,:, 1], markov_chain[:, t - 1])*(
                    np.dot(np.exp(-(np.dot(parameters[i, :, 2], markov_chain[:, t - 1]) * time_step[1:t+1])),h))
                #check if the integral is greater or equal than a drawn value of the exponential random variable with lambda=1
                default_indicator[t,i] = np.trapezoid(default_intensities[:t+1,i],dx=dt) >= np.random.exponential(1,1)[0]
                if default_indicator[t,i] ==1:
                    state_loss[t:,i] = np.random.gamma(scale=1, shape=5, size=1)[0]
    counting_process = np.sum(default_indicator, axis=1)

    return  counting_process,default_indicator,state_loss,default_intensities

def MarkovChainSim(states_number,transition_matrix,time,initial_state):
    """
    :param states_number: number of state of the chain
    :param transition_matrix: probabilities of changing states given the state we are now
    :param time: vector time
    :param initial_state: initial state of the markov chain
    :return: the path of chain
    """
    states = np.zeros(shape=(states_number,len(time)))
    states[:,0] = initial_state
    for i in range(1,len(time)):

        z =  np.random.choice(size=1,p=transition_matrix[np.where(states[:,i-1]==1)[0][0],],a=np.array((0,1)))
        states[z, i] = 1
    return states

# STOCHASTIC SIMULATION ALGORITHM
def simulate_sampled_ctmc_onehot(maturity=10.0, dt=0.01, lambda_=1.0, X0=0):
    sample_times = np.arange(0, maturity + dt, dt)
    num_steps = len(sample_times)

    # Initialize a 2 x N array for one-hot encoding
    states = np.zeros((2, num_steps), dtype=int)

    current_state = X0
    next_jump_time = np.random.exponential(1 / lambda_)

    for i, t_sample in enumerate(sample_times):
        while t_sample >= next_jump_time:
            current_state = 1 - current_state  # Flip state
            next_jump_time += np.random.exponential(1 / lambda_)

        # Set one-hot vector
        if current_state == 0:  # Good
            states[:, i] = [1, 0]
        else:  # Bad
            states[:, i] = [0, 1]

    return states

def ParametersGenerator(firm_number,states_number,lower_bound=0.001,upper_bound=0.009,state_difference = 0.1):
    """
    :param firm_number: number of firms in order to create n set of parameters
    :param states_number: number of states in order to define the rows of each firm's parameter, generally 2
    :param lower_bound:
    :param upper_bound:
    :param state_difference: difference of parameters between different states
    :return: number of firm * number of states * 3 parameters
    """
    param = np.zeros((firm_number, states_number, 3))
    for i in range(firm_number):
        h = 0
        for j in range(states_number):
            param[i, j, :] = np.random.choice(size=3, a=np.linspace(lower_bound, upper_bound, 10000)) + h
            h += np.random.choice(size=3, a=np.linspace(lower_bound, upper_bound, 10000))+ state_difference
    return param


def filtering_equation(initial_distribution,time,state_loss,states_number,eps,firm_number,parameters, default_indicator):
    transition_matrix = eps *  np.array([[-1 ,1],[1, -1]])
    filters = np.zeros(shape=(states_number,len(time)))
    filters[:,0] = initial_distribution
    dt =time[1]-time[0]
    dz = np.zeros(shape=(len(time), firm_number))
    diag = np.zeros(shape=(len(time),firm_number, 2, 2))
    for i in range(firm_number):
        g = np.where(state_loss[:, i] != 0)
        if len(g[0]) != 0 and g[0][0] < len(state_loss[:,0])-1 :
            dz[g[0][0] , i] = state_loss[g[0][0], i]
    for t in range(1,len(time)):
        b = np.zeros((2))
        c = np.zeros((2))

        time_step = abs(time - time[t])
        for i in range(firm_number):
            intensity = np.zeros(shape=(1, states_number))
            h = abs(np.hstack((0, np.diff(np.sum(np.delete(default_indicator, i, 1), axis=1)[:t]))))
            for n in range(states_number):
                intensity[0,n] = parameters[i, n, 0] + parameters[i,n, 1] * np.dot(np.exp(-parameters[i,n, 2] * time_step[1:t + 1]),h)
            diag[t,i,:,:] = np.diag(v=np.array((intensity[0, 0], intensity[0, 1])))

            if default_indicator[t,i]==0:
                u= [np.matmul(diag[t-l+1,i,:,:], filters[:, t - l].reshape(2, 1)) for l in range(t, 0, -1)]

                u.insert(0, np.array([[0], [0]]))
                c += np.trapezoid(u, dx=dt, axis=0).reshape(2)
            else:
                u = [np.matmul(diag[l, i, :, :], filters[:, l - 1].reshape(2, 1)) for l in range(1, np.where(default_indicator[:, i] == 1)[0][0] + 1)]
                u.insert(0, np.array([[0], [0]]))
                c += np.trapezoid(u,dx=dt,axis=0).reshape(2)

            if default_indicator[t, i] == 1:
                b1 = np.matmul(diag[np.where(default_indicator[:,i]==1)[0][0],i,:,:], filters[:, np.where(default_indicator[:,i]==1)[0][0]].reshape(2, 1))

                b += np.array((b1[0,0],b1[1,0])) * dz[np.where(default_indicator[:,i]==1)[0][0],i]

        a = np.trapezoid(np.hstack((np.array([[0],[0]]),transition_matrix @ filters[:,:t+1])), dx=dt, axis=1).reshape(2)
        print(t,a, b, c)
        filters[:, t] = filters[:, 0] + a + b - c

    filters = filters/np.sum(filters,axis=0)
    x = np.zeros((2,len(time)))
    for t in range(len(time)):
        if filters[0,t] > 0.5:
            x[:,t]= np.array((1,0))
        else:
            x[:, t] = np.array((0, 1))


    return filters,x


def lambda_hat(time,parameters,filter,default_indicator,firm_number):
    lamb_hat = np.zeros((len(time),firm_number))
    for i in range(firm_number):
        lamb_hat[0,i] = np.dot(
            filter[:,0],parameters[i,:,0]
        )
    for t in range(1,len(time)):
        for i in range(firm_number):
            if default_indicator[t,i]==0:
                h = abs(np.hstack((0, np.diff(np.sum(np.delete(default_indicator, i, 1), axis=1)[:t]))))
                time_step = abs(time - time[t])

                lamb_hat[t, i] = np.dot(filter[:, t], np.array((parameters[i, 0, 0] + parameters[i, 0, 1] * np.dot(
                    np.exp(-parameters[i, 0, 2] * time_step[1:t + 1]), h),
                                                                parameters[i, 1, 0] + parameters[i, 1, 1] * np.dot(
                                                                    np.exp(-parameters[i, 1, 2] * time_step[1:t + 1]),
                                                                    h))))
            else:
                lamb_hat[t,i]=0

    return lamb_hat

#"""
def path_simulation(time,maturity,firm_number,parameters,initial_rate=0.05,eps=2,n_simualtion=1000,initial_distribution=np.array(((0.65),(0.35))),shape=5,scale=100000000):
    rate=np.zeros((len(time),n_simualtion))
    rate[0,:]=initial_rate
    dI = np.zeros((len(time), n_simualtion))
    dt=time[1]-time[0]
    MC_sim = np.zeros((n_simualtion, 2, len(time)))
    for y in range(n_simualtion):
        MC_sim[y] = simulate_sampled_ctmc_onehot(maturity, dt=0.01, lambda_=0.5, X0=0)
    default_indicator = np.zeros((n_simualtion, len(time), firm_number))
    intensities = np.zeros((n_simualtion, len(time), firm_number))
    intensities_hat = np.zeros((n_simualtion, len(time), firm_number))
    intensities_hat2 = np.zeros((n_simualtion, len(time), firm_number))
    intensities[:, 0, :] = parameters[:, 0, 0]
    intensities_hat[:, 0, :] = initial_distribution @ np.array([[parameters[:, 0, 0]], [parameters[:, 1, 0]]]).reshape(2, firm_number)
    intensities_hat2[:, 0, :] = initial_distribution @ np.array([[parameters[:, 0, 0]], [parameters[:, 1, 0]]]).reshape(2, firm_number)
    k=1 # mean reverson velocity
    sigma=0.1 # volatility of cir process for the rates
    c1 = np.zeros((len(time), n_simualtion))
    c2 = np.zeros((len(time), n_simualtion))
    state_loss = np.zeros((n_simualtion, len(time), firm_number))
    filters = np.zeros((n_simualtion,2,len(time)))
    filters[:,:,0] = initial_distribution
    filters2 = np.zeros((n_simualtion, 2, len(time)))
    filters2[:, :, 0] = initial_distribution
    diag = np.zeros(shape=(n_simualtion,len(time), firm_number, 2, 2))
    dz = np.zeros(shape=(n_simualtion,len(time), firm_number))
    h=0

    for j in range(1, len(time)):
        if h < n_simualtion * len(time) * 0.1:
            print("|")
            print("0%")
        elif h < n_simualtion * len(time) * 0.2:
            print("|==")
            print("10%")
        elif h < n_simualtion * len(time) * 0.3:
            print("|====")
            print("20%")
        elif h < n_simualtion * len(time) * 0.4:
            print("|======")
            print("30%")
        elif h < n_simualtion * len(time) * 0.5:
            print("|========")
            print("40%")
        elif h < n_simualtion * len(time) * 0.6:
            print("|==========")
            print("50%")
        elif h < n_simualtion * len(time) * 0.7:
            print("|============")
            print("60%")
        elif h < n_simualtion * len(time) * 0.8:
            print("|==============")
            print("70%")
        elif h < n_simualtion * len(time) * 0.9:
            print("|================")
            print("80%")
        elif h < n_simualtion * len(time):
            print("|==================")
            print("90%")
        time_step = abs(time - time[j])
        rate[j,] = rate[j - 1,] + k* (np.array((0.03, 0.15)) @ MC_sim[:, :, j - 1].transpose() - rate[j - 1,]) * dt +(
                    sigma* np.sqrt(rate[j - 1,]) * np.random.randn(n_simualtion) * np.sqrt(dt))

        for n in range(n_simualtion):
            h+=1
            b = np.zeros((2))
            c = np.zeros((2))
            # compute intensities and generate defaults
            intensities[n, j, :] = np.dot(parameters[:, :, 0], MC_sim[n, :, j - 1]) + np.dot(parameters[:, :, 1],MC_sim[n, :, j - 1]) * ( np.exp(
                                   np.multiply(-np.dot(parameters[:, :, 2], MC_sim[n, :, j - 1]).reshape(3, 1),
                                   abs(time - time[j])[1:j + 1])) @ abs( np.hstack((0, np.diff(np.sum(default_indicator[n,], axis=1)[:j])))).transpose())
            intensities[n,j,:] = np.multiply(intensities[n,j,:],1*np.logical_not(default_indicator[n,j-1,:]))
            a= np.multiply(np.trapezoid(intensities[n, :j + 1, :], dx=dt,axis=0),1*np.logical_not(default_indicator[n,j-1,:])) >= np.random.exponential(1, 3)
            default_indicator[n,j,:] = abs(default_indicator[n,j-1,:]-a)

            if len(np.where((default_indicator[n,j,:]-default_indicator[n,j-1,:])==1)[0]):
                state_loss[n,j:,np.where((default_indicator[n,j,:]-default_indicator[n,j-1,:])==1)[0][0]] = scale*np.random.weibull(shape, 1)[0]

            # filter using only jumps
            for i in range(firm_number):
                g = np.where(state_loss[n,:, i] != 0)
                if len(g[0]) != 0 and g[0][0] < len(state_loss[n,:, 0]) - 1:
                    dz[n,g[0][0], i] = state_loss[n,g[0][0], i]

            for i in range(firm_number):
                intensity = np.zeros(shape=(1,2))

                for m in range(2):
                    intensity[0, m] = parameters[i, m, 0] + parameters[i, m, 1] * np.dot(
                        np.exp(-parameters[i, m, 2] * time_step[1:j + 1]),
                        abs(np.hstack((0, np.diff(np.sum(np.delete(default_indicator[n,], i, 1), axis=1)[:j])))))
                diag[n,j, i, :, :] = np.diag(v=np.array((intensity[0, 0], intensity[0, 1])))

                if default_indicator[n,j, i] == 0:
                    u = [np.matmul(diag[n,j - l + 1, i, :, :], filters[n,:, j - l].reshape(2, 1)) for l in range(j, 0, -1)]
                    u.insert(0, np.array([[0], [0]]))
                    c += np.trapezoid(u, dx=dt, axis=0).reshape(2)
                else:
                    u = [np.matmul(diag[n,l, i, :, :], filters[n,:, l - 1].reshape(2, 1)) for l in range(1, np.where(default_indicator[n,:, i] == 1)[0][0] + 1)]
                    u.insert(0, np.array([[0], [0]]))
                    c += np.trapezoid(u, dx=dt, axis=0).reshape(2)

                if default_indicator[n,j, i] == 1:

                    b1 = np.matmul(diag[n,np.where(default_indicator[n,:, i] == 1)[0][0], i, :, :],
                                   filters[n,:, np.where(default_indicator[n,:, i] == 1)[0][0]].reshape(2, 1))

                    b += np.array((b1[0, 0], b1[1, 0])) * dz[n,np.where(default_indicator[n,:, i] == 1)[0][0], i]


            a = np.trapezoid(np.hstack((np.array([[0], [0]]), eps *  np.array([[-1 ,1],[1, -1]]) @ filters[n,:, :j + 1])), dx=dt,axis=1).reshape(2)

            filters[n,:,j]= filters[n,:,0] + a - c + b

            # filter using only rates
            dI[j,n] =  (rate[j,n]-rate[j-1,n])/(sigma*np.sqrt(rate[j,n])) - dt*((0.03*filters2[n,0,j-1]+0.15*filters2[n,1,j-1]-rate[j,n])/(sigma*np.sqrt(rate[j,n])))
            c1[j,n] = (k * filters2[n,0,j-1]*(0.03-(0.03*filters2[n,0,j-1]+0.15*filters2[n,1,j-1])))/(sigma*np.sqrt(rate[j,n]))
            c2[j,n] = (k * filters2[n,1,j-1]*(0.15-(0.03*filters2[n,0,j-1]+0.15*filters2[n,1,j-1])))/(sigma*np.sqrt(rate[j,n]))

            filters2[n, 0, j] = (filters2[n, 0, 0] + np.trapezoid(np.hstack((np.array([0]), -eps * filters2[n,0, :j ])), dx=dt) +
                                 np.trapezoid(np.hstack((np.array([0]), eps * filters2[n, 1, :j ])), dx=dt) + dI[:j,n] @ c1[:j,n] )
            filters2[n, 1, j] = (filters2[n, 1, 0] + np.trapezoid(np.hstack((np.array([0]), eps * filters2[n, 0, :j ])), dx=dt)
                                 + np.trapezoid(np.hstack((np.array([0]), -eps * filters2[n, 1, :j ])), dx=dt) + dI[:j,n] @ c2[:j,n] )
            if filters2[n,0,j] > 1:
                filters2[n,0,j]=1
            elif filters2[n,0,j] < 0:
                filters2[n,0,j]=0
            if filters2[n, 1, j] > 1:
                filters2[n, 1, j] = 1
            elif filters2[n, 1, j] < 0:
                filters2[n, 1, j] = 0

            # default intensities under partial information using only defaults
            intensities_hat[n,j,:] = (filters[n,:,j]/np.sum(filters[n,:,j],axis=0)) @ np.array([[parameters[:, 0, 0] + parameters[:, 0, 1] * (
                 np.exp(-np.multiply(parameters[:, 0, 2].reshape(3, 1), abs(time - time[j])[1:j+1])) @ abs(
                 np.hstack((0, np.diff(np.sum(default_indicator[n,], axis=1)[:j])))).transpose())],[parameters[:, 1, 0] + parameters[:, 1, 1] * (
                 np.exp(-np.multiply(parameters[:, 1, 2].reshape(3, 1), abs(time - time[j])[1:j+1])) @ abs(
                 np.hstack((0, np.diff(np.sum(default_indicator[n,], axis=1)[:j])))).transpose())]]).reshape(2,firm_number)
            intensities_hat[n,j,:] = np.multiply(intensities_hat[n,j,:],1*np.logical_not(default_indicator[n,j-1,:]))

            # default intensities under partial information using only rates
            intensities_hat2[n, j, :] = filters2[n, :, j]  @ np.array([[parameters[:, 0, 0] + parameters[:, 0, 1] * (
                 np.exp(-np.multiply(parameters[:, 0, 2].reshape(3, 1), abs(time - time[j])[1:j + 1])) @ abs(
                 np.hstack((0, np.diff(np.sum(default_indicator[n,], axis=1)[:j])))).transpose())], [parameters[:, 1, 0] + parameters[:, 1, 1] * (
                 np.exp(-np.multiply(parameters[:, 1, 2].reshape(3, 1), abs(time - time[j])[1:j + 1])) @ abs(
                 np.hstack((0, np.diff(np.sum(default_indicator[n,], axis=1)[:j])))).transpose())]]).reshape(2,firm_number)
            intensities_hat2[n,j,:] = np.multiply(intensities_hat2[n, j, :],1 * np.logical_not(default_indicator[n, j - 1, :]))

    return intensities,intensities_hat,intensities_hat2,filters/np.sum(filters,axis=1).reshape(n_simualtion,1,len(time)),filters2,MC_sim,state_loss,default_indicator,np.sum(default_indicator,axis=2).reshape(n_simualtion,len(time),1),rate

#"""


"""
def path_simulation(time, transition_matrix, firm_number, parameters, initial_rate=0.05, eps=2, n_simualtion=1000,
                    initial_distribution=np.array(((0.65), (0.35))), shape=5, scale=100000000):

    rate = np.zeros((len(time), n_simualtion))
    rate[0, :] = initial_rate
    dI = np.zeros((len(time), n_simualtion))
    dt = time[1] - time[0]
    MC_sim = np.zeros((n_simualtion, 2, len(time)))

    for y in range(n_simualtion):
        MC_sim[y] = MarkovChainSim(2, transition_matrix, time, np.array((1, 0)))

    default_indicator = np.zeros((n_simualtion, len(time), firm_number))
    intensities = np.zeros((n_simualtion, len(time), firm_number))
    intensities_hat = np.zeros((n_simualtion, len(time), firm_number))
    intensities_hat2 = np.zeros((n_simualtion, len(time), firm_number))

    intensities[:, 0, :] = parameters[:, 0, 0]
    intensities_hat[:, 0, :] = initial_distribution @ np.array([[parameters[:, 0, 0]], [parameters[:, 1, 0]]]).reshape(2, firm_number)
    intensities_hat2[:, 0, :] = initial_distribution @ np.array([[parameters[:, 0, 0]], [parameters[:, 1, 0]]]).reshape(2, firm_number)

    k = 1
    sigma = 0.08
    c1 = np.zeros((len(time), n_simualtion))
    c2 = np.zeros((len(time), n_simualtion))
    state_loss = np.zeros((n_simualtion, len(time), firm_number))

    filters = np.zeros((n_simualtion, 2, len(time)))
    filters[:, :, 0] = initial_distribution
    filters2 = np.zeros((n_simualtion, 2, len(time)))
    filters2[:, :, 0] = initial_distribution

    diag = np.zeros(shape=(n_simualtion, len(time), firm_number, 2, 2))
    dz = np.zeros(shape=(n_simualtion, len(time), firm_number))

    # tqdm progress bar
    progress_bar = tqdm(total=(n_simualtion * (len(time) - 1)), desc="Simulating paths")

    for j in range(1, len(time)):
        time_step = abs(time - time[j])
        rate[j, ] = rate[j - 1, ] + k * (np.array((0.03, 0.18)) @ MC_sim[:, :, j - 1].T - rate[j - 1, ]) * dt + (
                sigma * np.sqrt(rate[j - 1, ]) * np.random.randn(n_simualtion) * np.sqrt(dt))

        for n in range(n_simualtion):
            b = np.zeros((2))
            c = np.zeros((2))

            intensities[n, j, :] = np.dot(parameters[:, :, 0], MC_sim[n, :, j - 1]) + np.dot(parameters[:, :, 1], MC_sim[n, :, j - 1]) * (
                np.exp(-np.dot(parameters[:, :, 2], MC_sim[n, :, j - 1]).reshape(3, 1) * time_step[1:j + 1]) @
                abs(np.hstack((0, np.diff(np.sum(default_indicator[n, ], axis=1)[:j])))).T)
            intensities[n, j, :] *= 1 * np.logical_not(default_indicator[n, j - 1, :])

            a = np.trapezoid(intensities[n, :j + 1, :], dx=dt, axis=0) >= np.random.exponential(1, 3)
            default_indicator[n, j, :] = abs(default_indicator[n, j - 1, :] - a)

            if np.any(default_indicator[n, j, :] - default_indicator[n, j - 1, :] == 1):
                firm_idx = np.where(default_indicator[n, j, :] - default_indicator[n, j - 1, :] == 1)[0][0]
                state_loss[n, j:, firm_idx] = scale * np.random.weibull(shape)

            for i in range(firm_number):
                g = np.where(state_loss[n, :, i] != 0)
                if len(g[0]) != 0 and g[0][0] < len(state_loss[n, :, 0]) - 1:
                    dz[n, g[0][0], i] = state_loss[n, g[0][0], i]

            for i in range(firm_number):
                intensity = np.zeros((1, 2))
                for m in range(2):
                    intensity[0, m] = parameters[i, m, 0] + parameters[i, m, 1] * np.dot(
                        np.exp(-parameters[i, m, 2] * time_step[1:j + 1]),
                        abs(np.hstack((0, np.diff(np.sum(np.delete(default_indicator[n, ], i, 1), axis=1)[:j])))))
                diag[n, j, i, :, :] = np.diag(np.array((intensity[0, 0], intensity[0, 1])))

                if default_indicator[n, j, i] == 0:
                    u = [diag[n, j - l + 1, i, :, :] @ filters[n, :, j - l].reshape(2, 1) for l in range(j, 0, -1)]
                    u.insert(0, np.array([[0], [0]]))
                    c += np.trapezoid(u, dx=dt, axis=0).reshape(2)
                else:
                    u = [diag[n, l, i, :, :] @ filters[n, :, l - 1].reshape(2, 1)
                         for l in range(1, np.where(default_indicator[n, :, i] == 1)[0][0] + 1)]
                    u.insert(0, np.array([[0], [0]]))
                    c += np.trapezoid(u, dx=dt, axis=0).reshape(2)

                if default_indicator[n, j, i] == 1:
                    idx = np.where(default_indicator[n, :, i] == 1)[0][0]
                    b1 = diag[n, idx, i, :, :] @ filters[n, :, idx].reshape(2, 1)
                    b += np.array((b1[0, 0], b1[1, 0])) * dz[n, idx, i]

            a = np.trapezoid(np.hstack((np.array([[0], [0]]), eps * np.array([[-1, 1], [1, -1]]) @ filters[n, :, :j + 1])), dx=dt, axis=1).reshape(2)
            filters[n, :, j] = filters[n, :, 0] + a - c + b

            dI[j, n] = (rate[j, n] - rate[j - 1, n]) / (sigma * np.sqrt(rate[j, n])) - dt * (
                    (0.05 * filters2[n, 0, j - 1] + 0.15 * filters2[n, 1, j - 1] - rate[j, n]) / (sigma * np.sqrt(rate[j, n])))

            c1[j, n] = (k * filters2[n, 0, j - 1] * (0.05 - (0.05 * filters2[n, 0, j - 1] + 0.15 * filters2[n, 1, j - 1]))) / (
                sigma * np.sqrt(rate[j, n]))
            c2[j, n] = (k * filters2[n, 1, j - 1] * (0.15 - (0.05 * filters2[n, 0, j - 1] + 0.15 * filters2[n, 1, j - 1]))) / (
                sigma * np.sqrt(rate[j, n]))

            filters2[n, 0, j] = filters2[n, 0, 0] + np.trapezoid(np.hstack((np.array([0]), -eps * filters2[n, 0, :j])), dx=dt) + \
                                np.trapezoid(np.hstack((np.array([0]), eps * filters2[n, 1, :j])), dx=dt) + dI[:j, n] @ c1[:j, n]
            filters2[n, 1, j] = filters2[n, 1, 0] + np.trapezoid(np.hstack((np.array([0]), eps * filters2[n, 0, :j])), dx=dt) + \
                                np.trapezoid(np.hstack((np.array([0]), -eps * filters2[n, 1, :j])), dx=dt) + dI[:j, n] @ c2[:j, n]

            filters2[n, :, j] = np.clip(filters2[n, :, j], 0, 1)

            filters_sum = np.sum(filters[n, :, j])
            if filters_sum > 0:
                filters[n, :, j] /= filters_sum

            intensities_hat[n, j, :] = filters[n, :, j] @ np.array([
                parameters[:, 0, 0] + parameters[:, 0, 1] * (
                    np.exp(-parameters[:, 0, 2].reshape(3, 1) * time_step[1:j + 1]) @
                    abs(np.hstack((0, np.diff(np.sum(default_indicator[n, ], axis=1)[:j])))).T),
                parameters[:, 1, 0] + parameters[:, 1, 1] * (
                    np.exp(-parameters[:, 1, 2].reshape(3, 1) * time_step[1:j + 1]) @
                    abs(np.hstack((0, np.diff(np.sum(default_indicator[n, ], axis=1)[:j])))).T)
            ]).reshape(2, firm_number)
            intensities_hat[n, j, :] *= 1 * np.logical_not(default_indicator[n, j - 1, :])

            intensities_hat2[n, j, :] = filters2[n, :, j] @ np.array([
                parameters[:, 0, 0] + parameters[:, 0, 1] * (
                    np.exp(-parameters[:, 0, 2].reshape(3, 1) * time_step[1:j + 1]) @
                    abs(np.hstack((0, np.diff(np.sum(default_indicator[n, ], axis=1)[:j])))).T),
                parameters[:, 1, 0] + parameters[:, 1, 1] * (
                    np.exp(-parameters[:, 1, 2].reshape(3, 1) * time_step[1:j + 1]) @
                    abs(np.hstack((0, np.diff(np.sum(default_indicator[n, ], axis=1)[:j])))).T)
            ]).reshape(2, firm_number)
            intensities_hat2[n, j, :] *= 1 * np.logical_not(default_indicator[n, j - 1, :])

            progress_bar.update(1)

    progress_bar.close()

    return intensities, intensities_hat, intensities_hat2, filters, filters2, MC_sim, state_loss, default_indicator, \
        np.sum(default_indicator, axis=2).reshape(n_simualtion, len(time), 1), rate
"""

"""
def simulate_single_path(time, transition_matrix, firm_number, parameters, initial_rate, eps,
                         initial_distribution, shape, scale, k, sigma, dt):
    len_time = len(time)
    time_step = abs(time - time[:, None])

    # Initialize variables for one simulation
    rate = np.zeros(len_time)
    dI = np.zeros(len_time)
    MC_sim = MarkovChainSim(2, transition_matrix, time, np.array((1, 0)))

    default_indicator = np.zeros((len_time, firm_number))
    intensities = np.zeros((len_time, firm_number))
    intensities_hat = np.zeros((len_time, firm_number))
    intensities_hat2 = np.zeros((len_time, firm_number))

    state_loss = np.zeros((len_time, firm_number))
    dz = np.zeros((len_time, firm_number))
    diag = np.zeros((len_time, firm_number, 2, 2))

    filters = np.zeros((2, len_time))
    filters2 = np.zeros((2, len_time))
    filters[:, 0] = initial_distribution
    filters2[:, 0] = initial_distribution

    c1 = np.zeros(len_time)
    c2 = np.zeros(len_time)

    rate[0] = initial_rate
    intensities[0, :] = parameters[:, 0, 0]
    intensities_hat[0, :] = initial_distribution @ np.array([[parameters[:, 0, 0]], [parameters[:, 1, 0]]]).reshape(2, firm_number)
    intensities_hat2[0, :] = intensities_hat[0, :]

    for j in range(1, len_time):
        # Shorten names
        filt = filters[:, j-1]
        filt2 = filters2[:, j-1]

        # CIR short-rate dynamics
        rate[j] = rate[j - 1] + k * (np.array((0.03, 0.18)) @ MC_sim[:, j - 1] - rate[j - 1]) * dt + \
                  sigma * np.sqrt(rate[j - 1]) * np.random.randn() * np.sqrt(dt)

        # Intensities under complete info
        past_defaults = np.hstack((0, np.diff(np.sum(default_indicator[:j], axis=1))))
        for i in range(firm_number):
            λ = parameters[i, :, 0] + parameters[i, :, 1] * (
                np.exp(-parameters[i, :, 2].reshape(2, 1) * time_step[j, :j]) @ past_defaults[:j])
            intensities[j, i] = np.dot(MC_sim[:, j - 1], λ)
            if default_indicator[j - 1, i] == 1:
                intensities[j, i] = 0

        # Default times
        a = np.trapezoid(intensities[:j+1], dx=dt, axis=0) >= np.random.exponential(1, firm_number)
        default_indicator[j] = np.abs(default_indicator[j - 1] - a)

        # Losses and dz
        for i in range(firm_number):
            if default_indicator[j, i] - default_indicator[j - 1, i] == 1:
                state_loss[j:, i] = scale * np.random.weibull(shape)
                dz[j, i] = state_loss[j, i]

        # Filtering
        c = np.zeros(2)
        b = np.zeros(2)
        for i in range(firm_number):
            intensity = np.zeros((2,))
            for m in range(2):
                intensity[m] = parameters[i, m, 0] + parameters[i, m, 1] * (
                    np.exp(-parameters[i, m, 2] * time_step[j, :j]) @ past_defaults[:j])
            diag[j, i] = np.diag(intensity)

            u = [diag[j - l + 1, i] @ filters[:, j - l] for l in range(j, 0, -1)]
            u.insert(0, np.array([0, 0]))
            c += np.trapezoid(u, dx=dt, axis=0)

            if default_indicator[j, i] == 1:
                idx = j
                b1 = diag[idx, i] @ filters[:, idx]
                b += b1 * dz[idx, i]

        filters[:, j] = filters[:, 0] + \
                        np.trapezoid(np.hstack((np.array([[0], [0]]), eps * np.array([[-1, 1], [1, -1]]) @ filters[:, :j + 1])), dx=dt, axis=1) - c + b

        # Innovation process and alternative filter
        dI[j] = (rate[j] - rate[j - 1]) / (sigma * np.sqrt(rate[j])) - dt * (
                (0.05 * filt2[0] + 0.15 * filt2[1] - rate[j]) / (sigma * np.sqrt(rate[j])))

        c1[j] = (k * filt2[0] * (0.05 - (0.05 * filt2[0] + 0.15 * filt2[1]))) / (sigma * np.sqrt(rate[j]))
        c2[j] = (k * filt2[1] * (0.15 - (0.05 * filt2[0] + 0.15 * filt2[1]))) / (sigma * np.sqrt(rate[j]))

        filters2[0, j] = filters2[0, 0] - np.trapezoid(eps * filters2[0, :j], dx=dt) + \
                         np.trapezoid(eps * filters2[1, :j], dx=dt) + dI[:j] @ c1[:j]
        filters2[1, j] = filters2[1, 0] + np.trapezoid(eps * filters2[0, :j], dx=dt) - \
                         np.trapezoid(eps * filters2[1, :j], dx=dt) + dI[:j] @ c2[:j]

        filters2[:, j] = np.clip(filters2[:, j], 0, 1)

        # Normalize
        if np.sum(filters[:, j]) > 0:
            filters[:, j] /= np.sum(filters[:, j])

        # Intensity estimates
        λ_0 = parameters[:, 0, 0] + parameters[:, 0, 1] * (
            np.exp(-parameters[:, 0, 2].reshape(firm_number, 1) * time_step[j, :j]) @ past_defaults[:j])
        λ_1 = parameters[:, 1, 0] + parameters[:, 1, 1] * (
            np.exp(-parameters[:, 1, 2].reshape(firm_number, 1) * time_step[j, :j]) @ past_defaults[:j])

        intensities_hat[j, :] = filters[:, j] @ np.vstack((λ_0, λ_1))
        intensities_hat2[j, :] = filters2[:, j] @ np.vstack((λ_0, λ_1))

        for arr in (intensities_hat[j, :], intensities_hat2[j, :]):
            arr *= np.logical_not(default_indicator[j - 1, :])

    return intensities, intensities_hat, intensities_hat2, filters, filters2, MC_sim, state_loss, default_indicator, \
           np.sum(default_indicator, axis=1).reshape(len(time), 1), rate

def path_simulation_parallel(time, transition_matrix, firm_number, parameters, initial_rate=0.05, eps=2,
                             n_simulation=1000, initial_distribution=np.array([0.65, 0.35]),
                             shape=5, scale=1e8):

    dt = time[1] - time[0]
    k = 1
    sigma = 0.08

    results = Parallel(n_jobs=-1)(
        delayed(simulate_single_path)(n, time, transition_matrix, firm_number, parameters, initial_rate, eps,
                                      initial_distribution, shape, scale, k, sigma, dt)
        for n in tqdm(range(n_simulation), desc="Parallel Simulations"))

    # Unpack results into arrays
    unpacked = list(zip(*results))
    return tuple(np.stack(arr) for arr in unpacked)
"""



def log_l(parameters,markov_chain, time,firm_number,default_indicator):
    parameters = parameters.reshape(firm_number, 2, 3)
    integral_term = 0
    log_likelihood = 0
    dt = time[1] - time[0]
    for i in range(firm_number):
        if np.where(default_indicator[:, i] == 1)[0].any():

            tstar = np.where(default_indicator[:, i] == 1)[0][0]
            h = abs(np.hstack((0, np.diff(np.sum(np.delete(default_indicator, i, 1), axis=1)[:tstar]))))
            time_step = abs(time - time[tstar])

            log_likelihood +=  np.log(np.dot(parameters[i,:, 0], markov_chain[:, tstar - 1]) + np.dot(parameters[i,:, 1], markov_chain[:, tstar - 1])*(
                    np.dot(np.exp(-(np.dot(parameters[i, :, 2], markov_chain[:, tstar - 1]) * time_step[1:tstar+1])),h)))
            #print(np.log(np.dot(parameters[i,:, 0], markov_chain[:, tstar - 1]) + np.dot(parameters[i,:, 1], markov_chain[:, tstar - 1])*( np.dot(np.exp(-(np.dot(parameters[i, :, 2], markov_chain[:, tstar - 1]) * time_step[1:tstar+1])),h))))
            integrand = np.zeros((tstar + 1))
            for t in range(1, tstar + 1):
                h = abs(np.hstack((0, np.diff(np.sum(np.delete(default_indicator, i, 1), axis=1)[:t]))))
                time_step = abs(time - time[t])

                integrand[t] = np.dot(parameters[i,:, 0], markov_chain[:, t - 1]) + np.dot(parameters[i,:, 1], markov_chain[:, t - 1])*(
                    np.dot(np.exp(-(np.dot(parameters[i, :, 2], markov_chain[:, t - 1]) * time_step[1:t+1])),h))

        else:
            t = len(time) - 1

            h = abs(np.hstack((0, np.diff(np.sum(np.delete(default_indicator, i, 1), axis=1)[:t]))))
            time_step = abs(time - time[t])

            log_likelihood += np.log(np.dot(parameters[i, :, 0], markov_chain[:, t - 1]) + np.dot(parameters[i, :, 1], markov_chain[:, t - 1]) * (
                            np.dot(np.exp(-(np.dot(parameters[i, :, 2], markov_chain[:, t - 1]) * time_step[1:t + 1])), h)))
            #print(np.log(np.dot(parameters[i, :, 0], markov_chain[:, t - 1]) + np.dot(parameters[i, :, 1], markov_chain[:, t - 1]) *                np.dot(np.exp(-(np.dot(parameters[i, :, 2], markov_chain[:, t - 1]) * time_step[1:t + 1])), h))))
            integrand = np.zeros((len(time)))
            for t in range(1, len(time)):
                h = abs(np.hstack((0, np.diff(np.sum(np.delete(default_indicator, i, 1), axis=1)[:t]))))
                time_step = abs(time - time[t])
                integrand[t] = np.dot(parameters[i,:, 0], markov_chain[:, t - 1]) + np.dot(parameters[i,:, 1], markov_chain[:, t - 1])*(
                    np.dot(np.exp(-(np.dot(parameters[i, :, 2], markov_chain[:, t - 1]) * time_step[1:t+1])),h))
        #print(np.trapezoid(integrand, dx=dt))
        integral_term += np.trapezoid(integrand, dx=dt)

    return -(log_likelihood - integral_term)

def partial_log_l(parameters,filters, time,firm_number,default_indicator):
    parameters = parameters.reshape(firm_number, 2, 3)
    integral_term = 0
    log_likelihood = 0
    dt = time[1] - time[0]
    for i in range(firm_number):
        if np.where(default_indicator[:, i] == 1)[0].any():

            t0 = np.where(default_indicator[:, i] == 1)[0][0]
            integrand = np.zeros((t0+1))

            h = abs(np.hstack((0, np.diff(np.sum(np.delete(default_indicator, i, 1), axis=1)[:t0]))))
            time_step = abs(time - time[t0])

            log_likelihood += np.dot(filters[:, t0], np.log(np.array((parameters[i, 0, 0] + parameters[i, 0, 1] * np.dot(np.exp(-parameters[i, 0, 2] * time_step[1:t0 + 1]), h)
                              ,parameters[i, 1, 0] + parameters[i, 1, 1] * np.dot(np.exp(-parameters[i, 1, 2] * time_step[1:t0 + 1]), h)))))


            for t in range(1, t0 + 1):

                h = abs(np.hstack((0, np.diff(np.sum(np.delete(default_indicator, i, 1), axis=1)[:t]))))
                time_step = abs(time - time[t])
                integrand[t] = np.dot(filters[:, t],
                            np.array((parameters[i, 0, 0] + parameters[i, 0, 1] * np.dot(np.exp(-parameters[i, 0, 2] * time_step[1:t + 1]), h)
                            ,parameters[i, 1, 0] + parameters[i, 1, 1] * np.dot(np.exp(-parameters[i, 1, 2] * time_step[1:t + 1]), h))))

        else:
            t = len(time) - 1

            h = abs(np.hstack((0, np.diff(np.sum(np.delete(default_indicator, i, 1), axis=1)[:t]))))
            time_step = abs(time - time[t])

            log_likelihood += np.dot(filters[:, t], np.log(np.array((parameters[i, 0, 0] + parameters[i, 0, 1] * np.dot(
                            np.exp(-parameters[i, 0, 2] * time_step[ 1:t + 1]), h),parameters[i, 1, 0] + parameters[i, 1, 1] * np.dot(
                            np.exp(-parameters[i, 1, 2] * time_step[1:t + 1]), h)))))

            integrand = np.zeros((len(time)))
            for t in range(1, len(time)):

                h = abs(np.hstack((0, np.diff(np.sum(np.delete(default_indicator, i, 1), axis=1)[:t ]))))
                time_step = abs(time - time[t])
                integrand[t] = np.dot(filters[:, t], np.array((parameters[i, 0, 0] + parameters[i, 0, 1] * np.dot(
                    np.exp(-parameters[i, 0, 2] * time_step[1:t + 1]), h)
                                                               , parameters[i, 1, 0] + parameters[i, 1, 1] * np.dot(
                    np.exp(-parameters[i, 1, 2] * time_step[1:t + 1]), h))))


        integral_term += np.trapezoid(integrand, dx=dt)

    return -(log_likelihood-integral_term)

def likelihood(parameters,markov_chain, time,firm_number,default_indicator):
    parameters=parameters.reshape(firm_number,2,3)
    likeli = 1
    dt = time[1] - time[0]
    for i in range(firm_number):
        if np.where(default_indicator[:, i] == 1)[0].any():

            t0 = np.where(default_indicator[:, i] == 1)[0][0]
            integrand = np.zeros((t0 + 1))

            h = abs(np.hstack((0, np.diff(np.sum(np.delete(default_indicator, i, 1), axis=1)[:t0]))))
            time_step = abs(time - time[t0])

            intensity = np.dot(parameters[i, :, 0], markov_chain[:, t0 - 1]) + np.dot(parameters[i, :, 1],markov_chain[:,t0 - 1]) * (
                                         np.dot(np.exp(-(np.dot(parameters[i, :, 2], markov_chain[:, t0 - 1]) * time_step[1:t0 + 1])),h))

            for t in range(1, t0 + 1):
                h = abs(np.hstack((0, np.diff(np.sum(np.delete(default_indicator, i, 1), axis=1)[:t]))))
                time_step = abs(time - time[t])

                integrand[t] = np.dot(parameters[i, :, 0], markov_chain[:, t - 1]) + np.dot(parameters[i, :, 1],
                                                                                            markov_chain[:, t - 1]) * (
                                   np.dot(np.exp(
                                       -(np.dot(parameters[i, :, 2], markov_chain[:, t - 1]) * time_step[1:t + 1])), h))

        else:

            t = len(time)-1


            h = abs(np.hstack((0, np.diff(np.sum(np.delete(default_indicator, i, 1), axis=1)[:t]))))
            time_step = abs(time - time[t])

            intensity = np.dot(parameters[i, :, 0], markov_chain[:, t - 1]) + np.dot(parameters[i, :, 1],markov_chain[:,t - 1]) * (
                                         np.dot(np.exp(-(np.dot(parameters[i, :, 2], markov_chain[:, t - 1]) * time_step[1:t + 1])),h))

            integrand = np.zeros((len(time)))
            for t in range(1, len(time)):
                h = abs(np.hstack((0, np.diff(np.sum(np.delete(default_indicator, i, 1), axis=1)[:t]))))
                time_step = abs(time - time[t])
                integrand[t] = np.dot(parameters[i, :, 0], markov_chain[:, t - 1]) + np.dot(parameters[i, :, 1],
                                                                                            markov_chain[:, t - 1]) * (
                                   np.dot(np.exp(
                                       -(np.dot(parameters[i, :, 2], markov_chain[:, t - 1]) * time_step[1:t + 1])), h))

        likeli = likeli *  (intensity*
                   np.exp(-np.trapezoid(integrand, dx=dt)))
    return -likeli



