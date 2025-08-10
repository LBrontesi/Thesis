
import matplotlib.pyplot as plt
import time
from simulate_default import *


#np.random.seed(27243) #caso mc non cambia e azienda defaulta, filtro sovrastima
#np.random.seed(272433) #filtro sottostima and different losses with scale=1000000,shape=0.5
#np.random.seed(2752433) # 3 different losses with scale=1000000,shape=0.5
#np.random.seed(13411) # 2 losses really different but similar result with scale=1000000,shape=0.5
#np.random.seed(12223)
#np.random.seed(122234)
#np.random.seed(279993)
#np.random.seed(2793) # higher losses and higher filter
#np.random.seed(864)

""" DEFAULT SIMULATION"""
#"""
dt = 0.01
T= 3
t = np.arange(0,T+dt,dt)
n_firm=3
r_param = np.array((((0.0001,0.0002,10),(0.001,0.002,5)),
                    ((0.0009,0.0008,4),(0.03,0.04,2)),
                    ((0.003,0.004,2),(0.01,0.02,1))))
hr = np.array((0.6,0.4))
n_sim=1

inten,inten_hat,inten_hat2,filters,filters2,MC,z,ni,nt,r = path_simulation(t,T,n_firm,r_param,eps=2,n_simualtion=n_sim)

#"""

"""PLOT"""
#"""
fig0, axs = plt.subplots(2, 2)
axs[0, 0].plot(t,MC[0,1,:])
axs[0,0].set_yticks(np.arange(0,2))
axs[0,0].set_yticklabels(["Good State","Bad State"], rotation='horizontal', fontsize=9)
axs[0, 0].set_title('Markov Chain')
axs[0, 1].plot(t,nt[0,])
axs[0, 1].set_title('Counting Process')
axs[1, 0].plot(t,inten[0,])
axs[1, 0].set_title('Default Intensities')
axs[1, 1].plot(t,z[0,],label=["$Firm_1$","$Firm_2$","$Firm_3$"])
axs[1, 1].set_title('State Loss')
axs[1,1].legend(loc=0)
plt.show()

fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(t,MC[0,1,:])
axs[0, 0].set_yticks(np.arange(0,2))
axs[0, 0].set_yticklabels(["Good State","Bad State"], rotation='horizontal', fontsize=9)
axs[0, 0].set_title('Markov Chain')
axs[0, 1].plot(t,nt[0,])
axs[0, 1].set_title('Counting Process')
axs[1, 0].plot(t,filters[0,0,:],color="red",label= "$P(X_t = 0 | \mathcal{F}_t^Z)$")
axs[1, 0].plot(t,filters[0,1,:],color="black",label= "$P(X_t = 1 | \mathcal{F}_t^Z)$")
axs[1, 0].legend(loc=0)
axs[1, 0].set_title('Filter')
axs[1, 1].plot(t,z[0,],label=["$Firm_1$","$Firm_2$","$Firm_3$"])
axs[1, 1].set_title('State Loss')
axs[1, 1].legend(loc=0)
plt.show()

fig12= plt.figure(figsize=(10, 6))
gs = fig.add_gridspec(nrows=2, ncols=2)

# Top-left: Markov Chain
ax1 = fig12.add_subplot(gs[0, 0])
ax1.plot(t, MC[0, 1, :])
ax1.set_yticks(np.arange(0, 2))
ax1.set_yticklabels(["Good State", "Bad State"], rotation='horizontal', fontsize=9)
ax1.set_title('Markov Chain')

# Top-right: Interest Rate
ax2 = fig12.add_subplot(gs[0, 1])
ax2.plot(t, r)
ax2.set_title('Interest rate')

# Bottom: Filter (spanning both columns)
ax3 = fig12.add_subplot(gs[1, :])
ax3.plot(t, filters2[0, 0, :], color="red", label="$P(X_t = 0 \mid \mathcal{F}_t^R)$")
ax3.plot(t, filters2[0, 1, :], color="black", label="$P(X_t = 1 \mid \mathcal{F}_t^R)$")
ax3.legend(loc=0)
ax3.set_title('Filter')
plt.tight_layout()
plt.show()

fig1, axs = plt.subplots(2, 2)
axs[0, 0].plot(t,MC[0,1,:],label="Markov Chain")
axs[0,0].set_yticks(np.arange(0,2))
axs[0,0].set_yticklabels(["Good State","Bad State"], rotation='horizontal', fontsize=9)
axs[0, 0].set_title('Markov Chain')
axs[0, 1].plot(t,r)
axs[0, 1].set_title('Default Intensities Under Partial Information Using Defaults')

axs[1, 0].plot(t,filters2[0,0,:],color="red",label= "$P(X_t = 0 | \mathcal{F}_t^R)$")
axs[1, 0].plot(t,filters2[0,1,:],color="black",label= "$P(X_t = 1 | \mathcal{F}_t^R)$")
axs[1, 0].set_title('Default Intensities Under Complete Information')
axs[1, 1].plot(t,inten_hat2[0,])
axs[1, 1].set_title('Default Intensities Under Partial Information Using Rates')
axs[0,0].legend(loc=0)
axs[1,0].legend(loc=0)
plt.show()

fig12, axs = plt.subplots(2, 2)
axs[0, 0].plot(t,MC[0,1,:],label="Markov Chain")
axs[0,0].set_yticks(np.arange(0,2))
axs[0,0].set_yticklabels(["Good State","Bad State"], rotation='horizontal', fontsize=9)
axs[0, 0].set_title('Markov Chain')
axs[0, 1].plot(t,nt[0,])
axs[0, 1].set_title('Default Intensities Under Partial Information Using Defaults')

axs[1, 0].plot(t,filters[0,0,:],color="red",label= "$P(X_t = 0 | \mathcal{F}_t^Z)$")
axs[1, 0].plot(t,filters[0,1,:],color="black",label= "$P(X_t = 1 | \mathcal{F}_t^Z)$")
axs[1, 0].set_title('Default Intensities Under Complete Information')
axs[1, 1].plot(t,inten_hat[0,])
axs[1, 1].set_title('Default Intensities Under Partial Information Using Rates')
axs[0,0].legend(loc=0)
axs[1,0].legend(loc=0)
plt.show()
# consider theorem 1 of section 3 paper elliott
# last term of the filter make the 2 probabilities to converge to 0.5
# second term put the probability of the good state being above because of the intensity in the bad state to be higher
# first term gives the jump in case of default
#"""

"""ISOLATION OF THE EFFECTS"""
"""


r_param2 = np.copy(r_param)
r_param2[:,:,1]=0

r_param3 = np.copy(r_param)
r_param3[:,1,:]= r_param3[:,0,:]


inten2 = np.zeros((2,len(t),n_firm))
inten2[0,0,:]= r_param2[:,0,0]
inten2[1,0,:] = r_param3[:,0,0]
for j in range(1,len(t)):

    inten2[1, j, :] = np.dot(r_param3[:, :, 0], MC[0, :, j - 1]) + np.dot(r_param3[:, :, 1], MC[0, :, j - 1]) * (
        np.exp(np.multiply(-np.dot(r_param3[:, :, 2], MC[0, :, j - 1]).reshape(3, 1), abs(t - t[j])[1:j + 1])) @ abs(
        np.hstack((0, np.diff(np.sum(ni[0,], axis=1)[:j])))).transpose())

    inten2[0, j, :] = np.dot(r_param2[:, :, 0], MC[0, :, j - 1]) + np.dot(r_param2[:, :, 1], MC[0, :, j - 1]) * (
        np.exp(np.multiply(-np.dot(r_param2[:, :, 2], MC[0, :, j - 1]).reshape(3, 1), abs(t - t[j])[1:j + 1])) @ abs(
        np.hstack((0, np.diff(np.sum(ni[0,], axis=1)[:j])))).transpose())


inten2= np.multiply(inten2,1*np.logical_not(ni[0,]))


inten_hat2 = np.zeros((2,len(t),n_firm))
inten_hat2[0,0,:]=hr@np.array([[r_param2[:, 0, 0]],[r_param2[:, 1, 0]]]).reshape(2,3)
inten_hat2[1,0,:] =hr@np.array([[r_param3[:, 0, 0]],[r_param3[:, 1, 0]]]).reshape(2,3)

for j in range(1, len(t)):

    inten_hat2[1, j, :] = filters[0,:,j] @(np.array([[r_param3[:, 0, 0] + r_param3[:, 0, 1] * (
            np.exp(-np.multiply(r_param3[:, 0, 2].reshape(3, 1), abs(t - t[j])[1:j+1])) @ abs(
        np.hstack((0, np.diff(np.sum(ni[0,], axis=1)[:j])))).transpose())],[r_param3[:, 1, 0] + r_param3[:, 1, 1] * (
            np.exp(-np.multiply(r_param3[:, 1, 2].reshape(3, 1), abs(t - t[j])[1:j+1])) @ abs(
        np.hstack((0, np.diff(np.sum(ni[0,], axis=1)[:j])))).transpose())]]).reshape(2,3))
    inten_hat2[0, j, :] = filters[0,:,j] @(np.array([[r_param2[:, 0, 0] + r_param2[:, 0, 1] * (
            np.exp(-np.multiply(r_param2[:, 0, 2].reshape(3, 1), abs(t - t[j])[1:j+1])) @ abs(
        np.hstack((0, np.diff(np.sum(ni[0,], axis=1)[:j])))).transpose())],[r_param2[:, 1, 0] + r_param2[:, 1, 1] * (
            np.exp(-np.multiply(r_param2[:, 1, 2].reshape(3, 1), abs(t - t[j])[1:j+1])) @ abs(
        np.hstack((0, np.diff(np.sum(ni[0,], axis=1)[:j])))).transpose())]]).reshape(2,3))

inten_hat2= np.multiply(inten_hat2,1*np.logical_not(ni[0,]))

fig2, axs = plt.subplots(2, 2)

axs[0, 0].plot(t,MC[0,1,:])
axs[0,0].set_yticks(np.arange(0,2))
axs[0,0].set_yticklabels(["Good State","Bad State"], rotation='horizontal', fontsize=9)
axs[0, 0].set_title('Markov Chain')

axs[0, 1].plot(t,inten[0,])
axs[0, 1].set_title('Contagion and Frailty Effects')
axs[1, 0].plot(t,inten2[0,])
axs[1, 0].set_title('Frailty Effect')
axs[1, 1].plot(t,inten2[1,],label=["$Firm_1$","$Firm_2$","$Firm_3$"])
axs[1, 1].set_title('Contagion Effect')
axs[1,1].legend(loc=0)
plt.show()




#"""





""" DEFAULTABLE ZERO COUPON BOND/ COUPON BOND PRICE IN COMPLETE AND PARTIAL INFORMATION """
"""

n_firm=3
dt=0.01
T=1
t = np.arange(0,T+dt,dt)
r_param = np.array((((0.0001,0.0002,10),(0.001,0.002,5)),
                    ((0.0009,0.0008,4),(0.03,0.04,2)),
                    ((0.003,0.004,2),(0.01,0.02,1))))
rate = 0.08 #annual
F=100
C=8 #annual
# number of simulations needed to obtain correctly the result
n_sim= round((T/dt)**2)
#number of simulations to be fast
#n_sim=1000
start = time.process_time()
inten,inten_hat,inten_hat2,filters,filters2,MC_sim,z,ni,nt,r = path_simulation(t,T,n_firm,r_param,n_simualtion=n_sim)
print(f" Simulation takes {round((time.process_time() - start)/60,2)} minutes to complete")
print(np.shape(r))
LGD=0.8
print(np.shape(ni))
defaults= np.sum(ni[:,-1,:],axis=0)
print(f"First firm has defaulted {int(defaults[0])} times out of {n_sim} simulations")
print(f"Second firm has defaulted {int(defaults[1])} times out of {n_sim} simulations")
print(f"Third firm has defaulted {int(defaults[2])} times out of {n_sim} simulations")


# Needed to make the integreal, otherwise the faster the default, the higher the price
inten_1 = inten.copy()
inten_hat_1 = inten_hat.copy()
inten_hat2_1 = inten_hat2.copy()
for i in range(n_sim):
    for j in range(n_firm):
        if len(np.where(inten[i,:,j]==0)[0]):
            inten[i,np.where(inten[i,:,j]==0)[0],j] = 1000000000
for i in range(n_sim):
    for j in range(n_firm):
        if len(np.where(inten_hat[i,:,j]==0)[0]):
            inten_hat[i,np.where(inten_hat[i,:,j]==0)[0],j] = 1000000000
for i in range(n_sim):
    for j in range(n_firm):
        if len(np.where(inten_hat2[i,:,j]==0)[0]):
            inten_hat2[i,np.where(inten_hat2[i,:,j]==0)[0],j] = 1000000000

integrand=np.zeros((n_sim,len(t),n_firm))
for i in range(n_sim):
    for j in range(len(t)):
        integrand[i,j,:] = np.trapezoid(np.insert(inten[i,j:,],obj=0,values=0,axis=0),dx=dt,axis=0)

# price based on complete information and fixed rate
pz1=F*np.mean(np.multiply(np.transpose(np.tile(np.exp(-rate*(t[-1]-t)),(3,1))),np.exp(-integrand)),axis=0)


integrand1=np.zeros((n_sim,len(t),n_firm))
for i in range(n_sim):
    for j in range(len(t)):
        integrand1[i,j,:] = np.trapezoid(np.insert(inten_hat[i,j:,],obj=0,values=0,axis=0),dx=dt,axis=0)

# price based on partial information ( defaults) and fixed rat
pz2=F*np.mean(np.multiply(np.transpose(np.tile(np.exp(-rate*(t[-1]-t)),(3,1))),np.exp(-integrand1)),axis=0)

#integral over the rates because now they are not fixed, no more exp(-r*(T-t))
integrand2=np.zeros((len(t),n_sim))
for j in range(len(t)):
    integrand2[j,:] = np.trapezoid(np.insert(r[j:],obj=0,values=0,axis=0),dx=dt,axis=0)

#price in complete info and stochastic rates
discount=np.zeros((n_sim,len(t),n_firm))
for n in range(n_sim):
   discount[n,]=np.exp(-integrand2[:,n].reshape(len(t),1))*np.exp(-integrand[n,])

#price in complete info and stochastic rates
pz4=F*np.mean(discount,axis=0)

integrand3=np.zeros((n_sim,len(t),n_firm))
for i in range(n_sim):
    for j in range(len(t)):
        integrand3[i,j,:] = np.trapezoid(np.insert(inten_hat2[i,j:,],obj=0,values=0,axis=0),dx=dt,axis=0)


discount1=np.zeros((n_sim,len(t),n_firm))
for n in range(n_sim):
   discount1[n,]=np.exp(-integrand2[:,n].reshape(len(t),1))*np.exp(-integrand3[n,])

# price based on partial info (rate) and stochastic rates
pz5=F*np.mean(discount1,axis=0)


plt.plot(pz1,linestyle='-.')
plt.plot(pz2,linestyle=":")
plt.show()

plt.plot(pz4,linestyle='-.')
plt.plot(pz5,linestyle=":")
plt.show()



print(pz1)
#price of recovery of treasury
print(F*(1-LGD)*np.transpose(np.tile(np.exp(-rate*(t[-1]-t)),(3,1)))+F*LGD*np.mean(np.multiply(np.transpose(np.tile(np.exp(-rate*(t[-1]-t)),(3,1))),np.exp(-integrand)),axis=0))
P_rt_final_1 = F*np.mean( (1-LGD) * np.transpose(np.tile(np.exp(-rate*(t[-1]-t)),(3,1))) + LGD* np.multiply(np.transpose(np.tile(np.exp(-rate*(t[-1]-t)),(3,1))),np.exp(-integrand)),axis=0)
P_rt_final_2 = np.mean(F *(1-LGD) * np.transpose(np.tile(np.exp(-rate*(t[-1]-t)),(3,1))) + F*LGD* np.multiply(np.transpose(np.tile(np.exp(-rate*(t[-1]-t)),(3,1))),np.exp(-integrand1)),axis=0)
P_rt_final_3 = np.mean(F *(1-LGD) * np.tile(np.transpose(np.exp(-integrand2))[:, :, np.newaxis], (1, 1, 3)) + F*LGD* discount,axis=0)
P_rt_final_4 = np.mean(F *(1-LGD) * np.tile(np.transpose(np.exp(-integrand2))[:, :, np.newaxis], (1, 1, 3)) + F*LGD* discount1,axis=0)
print(P_rt_final_1)

plt.plot(P_rt_final_1,linestyle='-.')
plt.plot(P_rt_final_2,linestyle=':')
plt.show()

plt.plot(P_rt_final_3,linestyle='-.')
plt.plot(P_rt_final_4,linestyle=':')
plt.show()
#"""
""" CDS PRICING"""
#"""
from scipy.optimize import minimize_scalar
#np.random.seed(89)

freq = 2
n_firm = 3
dt = 0.01
T = 1
t = np.arange(0, T + dt, dt)

r_param = np.array([[(0.01, 0.02, 10), (0.1, 0.2, 5)],
                    [(0.04, 0.05, 4), (0.4, 0.5, 2)],
                    [(0.09, 0.08, 2), (0.9, 0.8, 1)]])

rate = 0.08  # constant annual rate for simplified pricing
F = 100
C = 8  # annual coupon
n_sim = 10
LGD = 0.6

start = time.process_time()
inten, inten_hat, inten_hat2, filters, filters2, MC_sim, z, ni, nt, r = path_simulation(t, T, n_firm, r_param, n_simualtion=n_sim)
print(f"Simulation took {round((time.process_time() - start)/60,2)} minutes to complete")

inten[inten == 0] = 1e-6
inten_hat[inten_hat == 0] = 1e-6
inten_hat2[inten_hat2 == 0] = 1e-6

# Precompute discount factors from stochastic rates
r_integrals = np.array([np.trapezoid(r[j:], dx=dt, axis=0) for j in range(len(t))])

discount_factors = np.exp(-r_integrals.T).T  # shape: (len(t), n_sim)

# Compute cumulative intensities
lambda_integrals = np.zeros((n_sim, len(t), n_firm))
for i in range(n_sim):
    lambda_integrals[i] = np.array([np.trapezoid(inten[i, j:], dx=dt, axis=0) for j in range(len(t))])

# Combine with interest rate discounting
discount = np.exp(-lambda_integrals) * discount_factors[:, :, None]

# Premium dates and matrix
payment_times = np.arange(1 / freq, T + 0.01, 1 / freq)
is_premium_time = np.isin(t, payment_times).astype(int)
t_row = t[:, None]
t_col = t[None, :]
premium_matrix = ((t_col > t_row) & is_premium_time[None, :]).astype(int)

fair_spread = np.zeros((n_sim, n_firm))

for n in range(n_sim):
    disc = discount[n]
    x = (1 / freq) * premium_matrix[0] * disc
    premium_leg = np.sum(np.exp(-rate * (t[-1] - t))[:, None] * x, axis=0)

    survival = np.exp(-np.cumsum(inten[n] * dt, axis=0))
    default_leg = LGD * np.trapezoid(inten[n] * survival * np.exp(-rate * (t[:, None] - t[0])), dx=dt, axis=0)

    fair_spread[n] = default_leg / np.maximum(premium_leg, 1e-6)

# Compute CDS prices at all times
p_cds_1 = np.zeros((n_sim, len(t), n_firm))

for n in range(n_sim):
    for j in range(len(t)):
        disc = discount[n]
        x = (fair_spread[n] / freq) * premium_matrix[j] * disc
        premium_leg = np.sum(np.exp(-rate * (t[-1] - t))[:, None] * x, axis=0)

        survival = np.array([np.exp(-np.cumsum(inten[n, j:, f] * dt)) for f in range(n_firm)]).T
        default_leg = LGD * np.trapezoid(inten[n, j:] * survival * np.exp(-rate * (t[j:] - t[j]))[:, None], dx=dt, axis=0)

        p_cds_1[n, j] = default_leg - premium_leg

p_cds_1 = np.mean(p_cds_1, axis=0)

# Plot results
plt.plot(p_cds_1)
plt.title("CDS Price Over Time")
plt.xlabel("Time Index")
plt.ylabel("Price")
plt.show()
