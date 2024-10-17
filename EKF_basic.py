#!/usr/bin/env python3
import numpy as np

class EKF_Discrete:
    def __init__(self, xhat0, P0, f, F, h, H, Q, R ):
        """
        Create a propagate and update step functions for 
        discrete basic EKF for the system below.

        stores most recent propogated/updated values for state and its covariance
        x_newp = f( x_old, u ) + w
        z      = h( x_newp ) + v 
        F = df/dx
        H = dh/dx
        where w and v are the process and measurement noise respectively
        :param f: Discrete dynamics update, function of (x, u, w).
        :param F: Partial derivative of f wrt x, function of (x, u, w).
        :param h: Measurement function of (x, v)
        :param H: Partial derivative of h wrt x, function of (x, v).
        :param Q: Expected covariance of process noise.
        :param R: Expected covariance of measurement noise.
        """
        self.x = xhat0
        self.P = P0

        self.f = f
        self.F = F
        self.h = h
        self.H = H
        self.Q = Q
        self.R = R
        self.delta = np.zeros((2,1))
    def propagate(self, uk):
        '''
        Propogation step of EKF:
        MAPS
        x - current state estimate
        uk - current control input
        P - covariance of current state estimate
        INTO ---->
        x_newp - state estimate at next time step 
        based on transition function (propogation)
        P_newp - covariance of state estimate at next 
        time step based on transition function
        WITH
        x_newp = f( xk, uk )
        P_newp = Fk( xk, uk ) * Pk * Fk( xk, uk )' + Q
        '''
        fk:np.ndarray = self.f(self.x, uk)
        Fk:np.ndarray = self.F(self.x, uk)

        self.x = fk
        self.P = Fk @ self.P @ Fk.T + self.Q
        return self.x, self.P

    def update(self, z):
        '''
        Update step of EKF
        MAPS
        x_newp - state estimate at next time step 
        based on transition function (propogation)
        P_newp - covariance of state estimate at next 
        time step based on transition function
        z      - recorded measurement
        INTO --->
        x_newu - state estimate at next time step 
        based on propagation and update (transition function and measurement feedback)
        P_newu - covariance of state estimate at next time step 
        based on propagation and update (transition function and measurement feedback)
        WITH
        z_exp = h( x_newp ) 
        kK = P_newp * Hk'/(Hk * P_newp * Hk' + R)
        x_newu = x_newp + Kk * (z - z_exp)
        P_newu = (I - Kk * Hk ) * self.newp
        '''
        # print(f'{ type( self.x )}')
        # print(f'{ self.x }')
        hk:np.ndarray = self.h( self.x )
        Hk:np.ndarray = self.H( self.x )

        z_current = z
        z_expected = hk
        z_difference = z_current - z_expected

        Kk = self.P @ Hk.T @ np.linalg.inv(Hk @ self.P @ Hk.T + self.R)
        jump = Kk @ z_difference
        euc_jump = np.sqrt( jump[0,0] ** 2 + jump[1,0] ** 2)
        yaw_jump = jump[2,0].item()
        self.delta = np.array([ 
                                [euc_jump], 
                                [yaw_jump]
                                ])
        self.x = self.x + jump 
        self.P = ( np.eye(self.P.shape[0]) - Kk @ Hk ) @ self.P

        return self.x, self.P
    
class EKF_Runc:
    def __init__(self, xhat0, P0, f, F, h, H, Q ):
        """
        Create a propagate and update step functions for 
        discrete basic EKF for the system below where R(k) is time dependent.

        stores most recent propogated/updated values for state and its covariance
        x_newp = f( x_old, u ) + w
        z      = h( x_newp ) + v 
        F = df/dx
        H = dh/dx
        where w and v are the process and measurement noise respectively
        :param f: Discrete dynamics update, function of (x, u, w).
        :param F: Partial derivative of f wrt x, function of (x, u, w).
        :param h: Measurement function of (x, v)
        :param H: Partial derivative of h wrt x, function of (x, v).
        :param R: Expected covariance of measurement noise.
        """
        self.x = xhat0
        self.P = P0

        self.f = f
        self.F = F
        self.h = h
        self.H = H
        self.Q = Q
        self.delta = np.zeros((2,1))
    def propagate(self, uk):
        '''
        Propogation step of EKF:
        MAPS
        x - current state estimate
        uk - current control input
        P  - current covariance of state estimate
        INTO ---->
        x_newp - state estimate at next time step 
        based on transition function (propogation)
        P_newp - covariance of state estimate at next 
        time step based on transition function
        WITH
        x_newp = f( xk, uk )
        P_newp = Fk( xk, uk ) * Pk * Fk( xk, uk )' + Q
        '''
        fk:np.ndarray = self.f(self.x, uk)
        Fk:np.ndarray = self.F(self.x, uk)

        self.x = fk
        self.P = Fk @ self.P @ Fk.T + self.Q
        return self.x, self.P

    def update(self, z, R):
        '''
        Update step of EKF
        MAPS
        x_newp - state estimate at next time step 
        based on transition function (propogation)
        P_newp - covariance of state estimate at next 
        time step based on transition function
        R      - expected measurement covariance at current time
        z      - recorded measurement
        INTO --->
        x_newu - state estimate at next time step 
        based on propagation and update (transition function and measurement feedback)
        P_newu - covariance of state estimate at next time step 
        based on propagation and update (transition function and measurement feedback)
        WITH
        z_exp = h( x_newp ) 
        kK = P_newp * Hk'/(Hk * P_newp * Hk' + R)
        x_newu = x_newp + Kk * (z - z_exp)
        P_newu = (I - Kk * Hk ) * self.newp
        '''
        # print(f'{ type( self.x )}')
        # print(f'{ self.x }')
        hk:np.ndarray = self.h( self.x )
        Hk:np.ndarray = self.H( self.x )

        z_current = z
        z_expected = hk
        z_difference = z_current - z_expected

        Kk = self.P @ Hk.T @ np.linalg.inv(Hk @ self.P @ Hk.T + R)
        jump = Kk @ z_difference
    #    euc_jump = np.sqrt( jump[0,0] ** 2 + jump[1,0] ** 2)
    #    yaw_jump = jump[2,0].item()
    #    self.delta = np.array([ 
    #                            [euc_jump], 
    #                            [yaw_jump]
    #                            ])
        self.x = self.x + jump 
        self.P = ( np.eye(self.P.shape[0]) - Kk @ Hk ) @ self.P

        return self.x, self.P
    
if __name__=="__main__":
    xhat0 = np.array( [ [0], [0] ])
    P0    = 1e-3*np.eye(2)
    def f(x,u):
        xnew = x + u
        return xnew

    def F(x,u):
        return np.eye(2)

    def h(x):
        return x

    def H(x):
        return np.eye(2)

    Q = 1e-3*np.eye(2)

    test_EKF = EKF_Runc(xhat0=xhat0, P0=P0, f=f, 
        F=F, h=h, H=H, Q = Q)
    u = np.array([ [1.0], [1.0] ])
    R = 1e-2*np.eye(2)
    test_EKF.propagate(uk=u)
    print(test_EKF.x)
    z = test_EKF.x + 1e-3*np.array([ [1], [1] ])
    test_EKF.update( z , R)
    print(test_EKF.x)