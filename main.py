from battery import Battery
from kalman import ExtendedKalmanFilter as EKF
from protocol import launch_experiment_protocol
import numpy as np
import math as m
import pandas as pd

def get_EKF(R0, R1, C1, std_dev, time_step):
    # initial state (SoC is intentionally set to a wrong value)
    # x = [[SoC], [RC voltage]]
    x = np.matrix([[1],\
                   [0.0]])

    exp_coeff = m.exp(-time_step/(C1*R1))
    
    # state transition model
    F = np.matrix([[1, 0        ],\
                   [0, exp_coeff]])

    # control-input model
    B = np.matrix([[-time_step/(Q_tot * 3600)],\
                   [ R1*(1-exp_coeff)]])

    # variance from std_dev
    var = std_dev ** 2

    # measurement noise
    R = var

    # state covariance
    P = np.matrix([[var, 0],\
                   [0, var]])

    # process noise covariance matrix
    Q = np.matrix([[var/50, 0],\
                   [0, var/50]])

    def HJacobian(x):
        return np.matrix([[battery_simulation.OCV_model.deriv(x[0,0]), -1]])

    def Hx(x):
        return battery_simulation.OCV_model(x[0,0]) - x[1,0]

    return EKF(x, F, B, P, Q, R, Hx, HJacobian)


def plot_everything(time, true_voltage, mes_voltage, true_SoC, estim_SoC, current,OCVEstim,temp):
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax1 = fig.add_subplot(511)
    ax2 = fig.add_subplot(512)
    ax3 = fig.add_subplot(513)
    ax4 = fig.add_subplot(514)
    ax5 = fig.add_subplot(515)
    # title, labels
    ax1.set_title('')    
    ax1.set_xlabel('Time / s')
    ax1.set_ylabel('voltage / V')
    ax2.set_xlabel('Time / s')
    ax2.set_ylabel('Soc')
    ax3.set_xlabel('Time / s')
    ax3.set_ylabel('Current / A')
    ax4.set_xlabel('Time / s')
    ax4.set_ylabel('OCV / V')

    ax1.plot(time, true_voltage, label="True voltage")
    ax1.plot(time, mes_voltage, label="Mesured voltage")
    ax2.plot(time, true_SoC, label="True SoC")
    ax2.plot(time, estim_SoC, label="Estimated SoC")
    ax3.plot(time, current, label="Current")
    ax4.plot(time,OCVEstim)
    ax5.plot(temp)
    ax1.legend()
    ax2.legend()
    ax3.legend()

    plt.show()


if __name__ == '__main__':
    data = pd.read_csv("data.csv")
    # data["total_power"] = (data["power_left"]+data["power_right"])/96
    # temp = data["total_power"].tolist()
    power = []
    ener = (data["total_energy"]*1000*3600)/(data["time_step"]*96)
    temp = ener.tolist()
    for i in temp:
        if i>0:
            power.append(i)
    # total capacity
    print(max(power))
    print(len(power)/len(temp) * 100)
    Q_tot = 22
    
    # Thevenin model values
    R0 = 0.062
    R1 = 0.01
    C1 = 3000
    
    # time period
    time_step = 50/1000

    battery_simulation = Battery(Q_tot, R0, R1, C1)

    # discharged battery
    # battery_simulation.actual_capacity = 0
    
    # measurement noise standard deviation
    std_dev = 0.015

    #get configured EKF
    Kf = get_EKF(R0, R1, C1, std_dev, time_step)

    time         = [0]
    true_SoC     = [battery_simulation.state_of_charge]
    estim_SoC    = [Kf.x[0,0]]
    true_voltage = [battery_simulation.voltage]
    mes_voltage  = [battery_simulation.voltage + np.random.normal(0,0.1,1)[0]]
    current      = [battery_simulation.current]
    OCVEstim     = [battery_simulation.OCV]
    def update_all(actual_current,instant_power):
        battery_simulation.current = actual_current
        battery_simulation.update(time_step)

        time.append(time[-1]+time_step)
        current.append(actual_current)

        true_voltage.append(battery_simulation.voltage)
        mes_voltage.append(battery_simulation.voltage + np.random.normal(0, std_dev, 1)[0])
        
        Kf.predict(u=actual_current)
        Kf.update(mes_voltage[-1] + R0 * actual_current)
        
        true_SoC.append(battery_simulation.state_of_charge)
        estim_SoC.append(Kf.x[0,0]) 
        OCVEstim.append(battery_simulation.OCV)
        
        return battery_simulation.voltage #mes_voltage[-1]
    
    # launch experiment
    launch_experiment_protocol(Q_tot, time_step, update_all,power)

    # plot stuff
    plot_everything(time, true_voltage, mes_voltage, true_SoC, estim_SoC, current,OCVEstim,temp)
