import numpy as np
import statistics
import matplotlib.pyplot as plt
"""
dx = [VcosT, VsinT, w];
A = 0, 0, -Vsin(T), 
    0, 0,  Vcos(T), 
    0, 0,        0,

B = cos(T), 0
    sin(T), 0
    0,      1
"""
num_state  = 3 # X Y Theta
num_inputs = 2 # forward and angular rates

class tbest:
    """
    The estimator class.
    """
    def __init__(self, dt, P, Q, R):
        self.dt = dt
        self.x_ii = np.array([
            [0], # X position
            [0], # Y position
            [0]  # Orientation
        ])

        self.estimate_error = np.array([
            [0], # X position
            [0], # Y position
            [0]  # Orientation
        ])

        self.jump_est = np.array([
            [0.], # euclidian jump distance
            [0.]  # jump in Orientation
        ])

        self.H = np.array([
            [1, 0, 0], # Measure X Position + Bias X
            [0, 1, 0], # Measure Y Position + Bias Y
            [0, 0, 1]  # Measure Orientation Directly
        ])

        self.F = np.eye(self.x_ii.shape[0])
        self.G = np.eye(self.x_ii.shape[0], num_inputs)
        self.x_ji = self.x_ii
        self.x_jj = self.x_ii

        self.P_ii = P 
        self.P_ji = self.P_ii
        self.P_jj = self.P_ii

        self.Q = Q
        self.R = R

        self.parked      = False #boolean indicator bot reaching final point
        
    def spinOnce(self, u_ii, z_jj, flag):
        self.propogate(u_ii)
        if flag:
            self.update(z_jj)
            self.x_ii = self.x_jj
            self.P_ii = self.P_jj
        else:
            self.x_ii = self.x_ji
            self.P_ii = self.P_ji
        
 
    def propogate(self, u_ii):
        """
        The prediction step.
        """
        A = np.array([
            [0, 0, -u_ii[0,0]*np.sin(self.x_ii[2,0])],
            [0, 0,  u_ii[0,0]*np.cos(self.x_ii[2,0])],
            [0, 0,  0,                              ],
        ])
        B = np.array([
            [np.cos(self.x_ii[2,0]), 0],
            [np.sin(self.x_ii[2,0]), 0], 
            [0,                      1], 
        ])
        self.F = np.eye(A.shape[0]) + A * self.dt  
        self.G = B * self.dt 

        dx_ji = np.array([
            [u_ii[0,0]*np.cos(self.x_ii[2,0])],
            [u_ii[0,0]*np.sin(self.x_ii[2,0])],
            [u_ii[1,0]],
        ])
        self.x_ji = dx_ji * self.dt + self.x_ii

        self.P_ji = self.F @ self.P_ii @ self.F.transpose() + self.Q
               
    def update(self, z_jj):
        """
        The measurement update step.
        """
        z_ji = self.x_ji[0:3,:]
        y_jj = z_jj - z_ji
        self.estimate_error = y_jj
        S_jj = self.H @ self.P_ji @ self.H.transpose() + self.R 
        K_jj = self.P_ji @ self.H.transpose() @ np.linalg.inv(S_jj)
        delta = K_jj @ y_jj
        dist_jump  = np.sqrt(delta[0,0].item() ** 2 + delta[1,0].item() ** 2)
        angle_jump = delta[2,0].item()
        self.jump_est = np.array([ 
                                [dist_jump], 
                                [angle_jump]
                                ])

        self.x_jj = self.x_ji + delta 
        self.P_jj = (np.eye(self.F.shape[0]) - K_jj @ self.H) @ self.P_ji

class tbtru:
    """
    The true system dynamics.
    """
    def __init__(self, Q, R, dt):
        self.Q = Q
        self.R = R
        self.dt = dt

        self.x = np.array([
            [0], # X Position
            [0], # Y Position
            [0]  # Orientation
        ])
        self.x_i = np.zeros((3,1))
        self.x_j = np.zeros((3,1))
        self.z   = np.zeros((3,1))

    def propogate(self, u_ii):
        """
        Step forward the dynamics with process noise.
        """
        self.x_i = self.x
        dx_i = np.array([
            [u_ii[0,0]*np.cos(self.x_i[2,0])],
            [u_ii[0,0]*np.sin(self.x_i[2,0])],
            [u_ii[1,0]]
        ])
        # self.x_j = dx_i*self.dt + self.x_i + np.array([np.diag(self.Q)]).transpose()
        process_noise = np.random.multivariate_normal(np.zeros(3), self.Q, 1).T
        self.x_j = dx_i*self.dt + self.x_i + process_noise

    def sample(self):
        """
        Take a measurement of the system.
        # """
        # NEW_R = np.array([[0.4,0,0],
        #                   [0, 0.4, 0],
        #                   [0,0,0.15]])
        # meas_noise = np.random.multivariate_normal(np.zeros( 3 ), NEW_R, 1).T
        meas_noise = np.random.multivariate_normal(np.zeros( 3 ), self.R, 1).T
        self.z = self.x + meas_noise
        # self.z = self.x + np.array([np.diag(self.R)]).transpose()
        return self.z

    def spinOnce(self, u_ii):
        self.propogate(u_ii)
        self.x = self.x_j
        
class ctrl:
    """
    The controller and path planner algorithm.
    """
    def __init__(self, dt, v, K, path_array, v_reg = True ):
        self.dt = dt
        self.V = v
        self.t = 0.
        self.K = K
        self.v_reg = v_reg
        self.u = np.zeros((2,1))
        self.path_index = 0
        self.path_array = path_array
        self.error_vector   = np.zeros( (3,1) )
        self.prev_yaw_error = 0.
        self.I_yaw_error    = 0.
        self.init_ind       = True # boolean describing if a new goal point ha just been selected
        self.dedt           = 0.
        self.arrival_times  = []
        self.distance_sum = 0.
        self.counter      = 0.
    
    def compute_error(self, x):
        """
        x - state in form [x, y, yaw]
        Compute the control action.
        """
        
        # x, y error and the euclidian error
        x_diff = self.current_point[0] - x[0,0]
        y_diff = self.current_point[1] - x[1,0]
        self.distance = np.sqrt(x_diff ** 2 + y_diff ** 2)

        # calcs for RMS error 
        self.distance_sum += self.distance
        self.counter += 1
        # yaw error
        current_yaw = x[2,0]
        desired_yaw = np.arctan2(y_diff, x_diff)

        yaw_error = desired_yaw - current_yaw

        # ensure yaw_error is within pi and -pi to prevent spinning
        if yaw_error >= np.pi:
            yaw_error -= 2*np.pi 
        if yaw_error <= -np.pi:
            yaw_error += 2*np.pi 

        self.error_vector = np.array([ 
                                     [x_diff], 
                                     [y_diff], 
                                     [yaw_error]
                                     ])

    def compute_command(self):
       
        yaw_error = self.error_vector[2,0]
        
        # update integral and derivateive terms

        if self.init_ind:
            # first spin since new point is chosen 
            # DO NOT UPDATE I AND dedt
            self.init_ind = False
        else:
            # it is not first spin since new target is chosen 
            # UPDATE I AND dedt
            self.dedt = (yaw_error - self.prev_yaw_error)/self.dt
            # print(f'dedt is {self.dedt}')
        self.I_yaw_error += yaw_error * self.dt

        kP = self.K[0].item()
        kI = self.K[1].item()
        kD = self.K[2].item()

        P_gain = kP*yaw_error
        I_gain = kI*self.I_yaw_error
        D_gain = kD*self.dedt 

        self.angular_gain_vector = np.array([ 
                                            [P_gain], 
                                            [I_gain], 
                                            [D_gain], 
                                            ])
        
        w = np.sum( self.angular_gain_vector ) 

        #checks for saturation
        if w >= 2.84:
            w = 2.84
        if w <= -2.84:
            w = -2.84
        
        if self.v_reg: # controller has capability to regulate lin speed
            # calculate linear
            if abs( yaw_error )>= np.pi/4:
                act_linear = float( 0 )
            else:
                act_linear = float( self.V / (np.pi/4) * \
                    (np.pi/4 - abs( yaw_error ) ) )
        else:
            # controller is unable to regulate speed
            act_linear = self.V 


        self.u = np.array([
            [act_linear],
            [w]
        ])

        # update previous yaw error for next time step. 
        self.prev_yaw_error = yaw_error
        
    def compute_trajectory(self):
        '''
        compute trajectory from point in path list
        '''
        # set current target
        self.current_point = self.path_array[:,self.path_index]

    def update(self):
        """
        Increment the time.
        """
        self.t += self.dt

    def spinOnce(self, x):
        self.compute_trajectory()
        self.compute_error(x)
        self.compute_command() # Need to add a state measurement here
        self.update()

    def reset(self):
        self.t = 0.
        self.u = np.zeros((2,1))
        self.path_index = 0
        self.error_vector   = np.zeros( (3,1) )
        self.prev_yaw_error = 0.
        self.I_yaw_error    = 0.
        self.init_ind       = True # boolean describing if a new goal point ha just been selected
        self.dedt           = 0.
        self.arrival_times  = []
        self.distance_sum = 0.
        self.counter      = 0.
    
    def check(self, tolerance, tbest_object):
    # Determines if we are close enough to intermediary point or final point
        if self.distance < tolerance: # close enough to current goal 
            # update boolean to show that it is the fist spin for new target point
            self.init_ind = True
            self.prev_yaw_error = 0. # reset previous yaw error to zero
            self.I_yaw_error    = 0. # reset integral error back to zero
            self.arrival_times.append(self.t) # add current time to list of arrival times
            # print(f'current time {self.t}')
            # print(f'current goal point index {self.path_index}')
            # print(f'current goal point {self.path_array[ :, self.path_index ]}')
            if np.array_equal(self.path_array[ :, self.path_index ]
                              , self.path_array[:, -1] ): #close enough to final goal
                tbest_object.parked = True
            else:
                # proceed to next point
                self.path_index += 1

class plotter:
    """
    Record information for plotting.
    """
    def __init__(self, path_array ):
       
        self.path_array = path_array
        self.t = []
        self.x = [] # The estimator state
        self.r = [] # The true state
        self.P = [] # The covariance of the estimator
        self.u = [] # The control inputs (radial and angular speed)
        self.e = []
        self.angular_gain = []
        self.estimate_error = []
        self.t_est = []
        self.jump = []
        self.count = 0 

    def spinOnce(self, x, t, r, P, u, e, 
                 angular_gain, est_error, 
                 t_est, flag, jump):
        
        self.record(x, t, r, P, u, e, 
                    angular_gain, est_error, 
                    t_est, flag, jump)
        self.update()

    def record(self, x, t, r, P, u, e, 
               angular_gain, est_error, 
               t_est, flag, jump):
        """
        Record given information.
        """
        self.x.append( x )
        self.t.append( t )
        self.r.append( r )
        self.P.append( P )
        self.u.append( u )
        self.e.append( e )
        self.angular_gain.append( angular_gain )
        
        if flag:
            self.estimate_error.append( est_error )
            self.t_est.append( t_est )
            self.jump.append(jump)

    def update(self):
        """
        Increment the record index.
        """
        self.count += 1

    def plot(self, arrival_times, debug = False):
        """
        Plot the stored information.
        """
        self.x = np.hstack( self.x )
        self.t = np.hstack( self.t )
        self.r = np.hstack( self.r )
        self.P = np.stack( self.P, axis=2 )
        self.u = np.hstack( self.u )
        self.e = np.hstack( self.e )
        self.estimate_error = np.hstack( self.estimate_error )
        self.angular_gain = np.hstack( self.angular_gain )
        self.arrival_times = np.array( arrival_times )
        self.jump = np.hstack( self.jump )
        zeros = np.zeros_like(self.arrival_times)

        if debug:
            # This is the plot of state 1 (X Position) estimation with truth, desired, and confidence interval
            plt.figure(1)
            plt.scatter(self.arrival_times, zeros, color = 'r',  label='Arrival Times')
            plt.plot(self.t, self.x[0,:],'b', linestyle = ':', label='estimate')
            plt.plot(self.t, self.r[0,:], 'k', label='truth')
            plt.plot(self.t, self.x[0,:]+3*np.sqrt(self.P[0,0,:]), 'r', label='confidence')
            plt.plot(self.t, self.x[0,:]-3*np.sqrt(self.P[0,0,:]), 'r', label='confidence')
            plt.xlabel('Time (s)')
            plt.ylabel('X Pose (m)')
            plt.legend(loc='upper right')
            plt.show()

            # Plot 1 but for state 2 (Y Position)
            plt.figure(2)
            plt.scatter(self.arrival_times, zeros, color = 'r',  label='Arrival Times')
            plt.plot(self.t, self.x[1,:], 'b', label='estimate')
            plt.plot(self.t, self.r[1,:], 'k', label='truth')
            plt.plot(self.t, self.x[1,:]+3*np.sqrt(self.P[1,1,:]), 'r', label='confidence')
            plt.plot(self.t, self.x[1,:]-3*np.sqrt(self.P[1,1,:]), 'r', label='confidence')
            plt.xlabel('Time (s)')
            plt.ylabel('Y Pose (m)')
            plt.legend(loc='upper right')
            plt.show()

            # Plot 1 but for state 3 (Theta)
            plt.figure(3)
            plt.scatter(self.arrival_times, zeros, color = 'r',  label='Arrival Times')
            plt.plot(self.t, self.x[2,:], 'b', label='estimate')
            plt.plot(self.t, self.r[2,:], 'k', label='truth')
            plt.plot(self.t, self.x[2,:]+3*np.sqrt(self.P[2,2,:]), 'r', label='confidence')
            plt.plot(self.t, self.x[2,:]-3*np.sqrt(self.P[2,2,:]), 'r', label='confidence')
            plt.xlabel('Time (s)')
            plt.ylabel('Theta (rad)')
            plt.legend(loc='upper right')
            plt.show()

            # XY plot showing the estimator, truth, and desired trajectories through time
            plt.figure(4)
            plt.plot(self.path_array[0,:], self.path_array[1,:], color = 'r', label = 'Desired Path' )
            plt.plot(self.x[0,:], self.x[1,:], color='b', linestyle = ':', linewidth = 4, label='estimate')
            plt.plot(self.r[0,:], self.r[1,:], color='k', label='truth')
            plt.xlabel('X Pose (m)')
            plt.ylabel('Y Pose (m)')
            plt.legend(loc='upper right')
            plt.xlim(-4,4)
            plt.ylim(-4,4)
            plt.show()

            # plot the desired radial speed
            plt.figure(5)
            plt.scatter(self.arrival_times, zeros, color = 'r',  label='Arrival Times')
            plt.plot(self.t, self.u[0,:], 'b', label='Desired radial speed')
            plt.xlabel('Time (s)')
            plt.ylabel('Radial speed (m/s)')
            plt.legend(loc='upper right')
            plt.show()

            # plot the desired angular velocity
            plt.figure(6)
            plt.scatter(self.arrival_times, zeros, color = 'r',  label='Arrival Times')
            plt.plot(self.t, self.u[1,:], 'r', linestyle =':', label='Desired angular velocity')
            plt.xlabel('Time (s)')
            plt.ylabel('Angular velocity (rad/s)')
            plt.legend(loc='upper right')
            plt.show()

            # plot the x error
            plt.figure(7)
            plt.scatter(self.arrival_times, zeros, color = 'r',  label='Arrival Times')
            plt.plot(self.t, self.e[0,:], 'b', label='tracking error in x position')
            plt.xlabel('Time (s)')
            plt.ylabel('Error (m)')
            plt.legend(loc='upper right')
            plt.show()

            # plot the y error
            plt.figure(8)
            plt.scatter(self.arrival_times, zeros, color = 'r',  label='Arrival Times')
            plt.plot(self.t, self.e[1,:], 'b', label='tracking error in y position')
            plt.xlabel('Time (s)')
            plt.ylabel('Error (m)')
            plt.legend(loc='upper right')
            plt.show()

            # plot the yaw error
            plt.figure(9)
            plt.scatter(self.arrival_times, zeros, color = 'r',  label='Arrival Times')
            plt.plot(self.t, self.e[2,:], 'b', label='tracking error in yaw')
            plt.xlabel('Time (s)')
            plt.ylabel('Error (rad)')
            plt.legend(loc='upper right')
            plt.show()

            # plot the PID terms
            plt.figure(10)
            plt.scatter(self.arrival_times, zeros, color = 'r',  label='Arrival Times')
            plt.plot(self.t, self.angular_gain[0,:], 'r', label='P Gain')
            plt.plot(self.t, self.angular_gain[1,:], 'g', linestyle = '--', label='I Gain')
            plt.plot(self.t, self.angular_gain[2,:], 'b', linestyle = '-.', label='D Gain')
            plt.xlabel('Time (s)')
            plt.ylabel('Control Gains (rad/s)')
            plt.legend(loc='upper right')
            plt.show()

            # plot the estimate error
            plt.figure(11)
            plt.scatter(self.arrival_times, zeros, color = 'r',  label='Arrival Times')
            plt.plot(self.t_est, self.estimate_error[0,:], 'r', label='x estimate error (m)')
            plt.plot(self.t_est, self.estimate_error[1,:], 'g', linestyle = '--', label='y estimate error (m)')
            plt.plot(self.t_est, self.estimate_error[2,:], 'b', linestyle = '-.', label='yaw estimate error (rad)')
            plt.xlabel('Time (s)')
            plt.ylabel('Estimate Error')
            plt.legend(loc='upper right')
            plt.show()

            # plot the jumps following an estimate update
            plt.figure(12)
            plt.scatter(self.arrival_times, zeros, color = 'r',  label='Arrival Times')
            plt.plot(self.t_est, self.jump[0,:], 'r', label='Distance jump (m)')
            plt.plot(self.t_est, self.jump[1,:], 'g', linestyle = '--', label='angle jump (rad)')
            plt.xlabel('Time (s)')
            plt.ylabel('Estimate jumps')
            plt.legend(loc='upper right')
            plt.show()

            plt.figure(13)
            plt.scatter(self.arrival_times, zeros, color = 'r',  label='Arrival Times')
            plt.plot(self.t, self.P[0,0,:], 'r', label='covariance')
            plt.xlabel('Time (s)')
            plt.ylabel('Covariance')
            plt.legend(loc='upper right')
            plt.show()
        else:
            # XY plot showing the estimator, truth, and desired trajectories through time
            plt.figure(1)
            plt.plot(self.path_array[0,:], self.path_array[1,:], color = 'r', marker = 'o', label = 'Desired Path' )
            plt.plot(self.x[0,:], self.x[1,:], color='b', linestyle = ':', linewidth = 2, label='estimate')
            plt.plot(self.r[0,:], self.r[1,:], color='k', label='truth')
            plt.xlabel('X Pose (m)')
            plt.ylabel('Y Pose (m)')
            plt.legend(loc='upper right')
            plt.xlim(-4.5,4.5)
            plt.ylim(-4.5,4.5)
            plt.show()

    def reset(self):
        self.t = []
        self.x = [] # The estimator state
        self.r = [] # The true state
        self.P = [] # The covariance of the estimator
        self.u = [] # The control inputs (radial and angular speed)
        self.e = []
        self.angular_gain = []
        self.estimate_error = []
        self.t_est = []
        self.jump = []
        self.count = 0 

if __name__=="__main__":
    
    # Some simulation constants
    dt = 1e-2    # time step
    P = (0.1**2)*np.eye(num_state)
    # Q = 1e-3*dt*np.eye(num_state) # process noise
    JUMPY = False

    if JUMPY:
        x_cp   = 1e-3 # x process covariance
        y_cp   = 1e-3 # y process covariance
        yaw_cp = 1e-3 # yaw process covariance
        Q       = dt * np.diag([x_cp, y_cp, yaw_cp])
        R_k = (1e-3)*np.eye(num_state) # measurement noise used in kalman gain of update step
        R_true = 1e-1*np.diag([0.4, 0.4, 0.15]) # true measurement noise
        dn = 15.     # measurement period
    else:
        x_cp   = 1e-4 # x process covariance
        y_cp   = 1e-4 # y process covariance
        yaw_cp = 1e-4 # yaw process covariance
        Q       = dt * np.diag([x_cp, y_cp, yaw_cp])
        R_k = (1e-5)*np.eye(num_state) # measurement noise for kalman gain
        R_true = R_k # true measurement noise equal to R_k in non jumpy case
        dn = 5.     # measurement period
    # CONTROL
    v = 0.2   # desired tbot radial speed (held fixed)
    kP = 5e-0  # P controller gain
    kI = 1e2  # I controller gain
    kD = 1e-2  # D controller gain
    v_reg = False # boolean to signal if we can control linear speed
    K  = np.array([ kP, kI, kD ])
    
    tolerance = 1e-1
    period = 2*np.pi # period of figure 8

    # Load the list of points from the .npy file
    figure_eight_points_array = np.load('figure_eight_points.npy', allow_pickle=True)
    path_array = figure_eight_points_array
    # path_array = np.array( [ 
                            # [6, 8], 
                            # [-6, 0] 
                            # ])
    # Convert the loaded array back to a list of NumPy arrays
    figure_eight_points_list = [np.array(point) for point in figure_eight_points_array]
        
    # Initialize the objects
    TBEST = tbest(dt, P, Q, R_k)
    TBTRU = tbtru(Q, R_true, dt)
    CTRL  = ctrl(dt, v, K, path_array, v_reg=v_reg)
    PLOT  = plotter(path_array)
    T_list = []
    RMS_list = []
    iters = 1
    for index in np.arange(iters):
        # Run through the simulation and retrieve needed information from each object 
        x = np.array([
                [0], # X position
                [0], # Y position
                [0]  # Orientation
                    ])
        
        z = np.array([
                [0], # X position
                [0], # Y position
                [0]  # Orientation
                    ])
        t = 0.      # current time value
        flag = False
        TBEST.parked = False
        CTRL.reset()
        # # reset plots unless it is the final spin
        # if abs( index - (iters - 1) ) > 1e-2:
        #     PLOT.reset()
        while not TBEST.parked:
            # Calculate the control action and error
            CTRL.spinOnce(x)
            e = CTRL.error_vector
            angular_gain_vector = CTRL.angular_gain_vector
            u = CTRL.u
            
            # Calculate the true state dynamics
            TBTRU.spinOnce(u)
            r = TBTRU.x
            
            # Measurement Block
            if (int(t/dt) % int(dn/dt) == 0):
                flag = True
                z = TBTRU.sample()
                estimate_error = TBEST.estimate_error
            # Calculate the estimator state dynamics
            TBEST.spinOnce(u,z,flag)
            x = TBEST.x_ii
            P = TBEST.P_ii

            # Record information for plots on first iter
            if index == 0:
                PLOT.spinOnce(x,t,r,P,u,e=e,
                            angular_gain=angular_gain_vector,
                            est_error=estimate_error,
                            t_est=t, flag=flag, 
                            jump=TBEST.jump_est)

            CTRL.check(tolerance=tolerance,tbest_object=TBEST)
            t += dt
            flag = False
        # calculate RMS from sum of distance errors
        RMS = np.sqrt(CTRL.distance_sum/CTRL.counter)
        
        # add to lists
        T_list.append( t )
        RMS_list.append( RMS )

    if iters > 1:
        stdev_lap_time = statistics.stdev(T_list)
        stdev_RMS = statistics.stdev(RMS_list)
        print(f'Standard deviation lap time is {stdev_lap_time}')
        print(f'Standard deviation RMS is {stdev_RMS}')

    mean_lap_time = statistics.mean(T_list)
    mean_RMS = statistics.mean(RMS_list)

    print(f'Mean lap time is {mean_lap_time}')
    print(f'Mean RMS is {mean_RMS}')

    # Plot
    PLOT.plot(CTRL.arrival_times)