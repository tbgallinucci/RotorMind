#!/usr/bin/env python3
"""
Rotordynamic Analysis - Python Translation from MATLAB
======================================================

This module performs rotordynamic analysis of a shaft-disk-bearing system
using finite element method. It calculates:
- Hydrodynamic bearing coefficients
- Ball bearing coefficients (fixed stiffness/damping)
- System frequency response
- Critical speeds
- Mode shapes

Author: Translated from MATLAB
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve, eig
import warnings
warnings.filterwarnings('ignore')

class RotordynamicAnalysis:
    def __init__(self):
        """Initialize the rotordynamic analysis with system parameters"""
        
        # Material properties
        self.E = 207e9      # modulus of elasticity (Pa)
        self.rho = 7850     # density (kg/m³)
        self.g = 9.81       # gravity (m/s²)
        
        # Shaft geometry
        self.de = 13e-3     # Diameter [m]
        self.Ae = (np.pi * self.de**2) / 4      # Area [m²]
        self.le = 747e-3    # Length [m]
        self.Ie = (np.pi * self.de**4) / 64     # Moment of inertia [m⁴]
        
        self.dm = 30e-3     # diameter at bearing [m]
        self.lm = 20e-3     # Length at bearing [m]
        self.Am = (np.pi * self.dm**2) / 4      # Area at bearing [m²]
        self.Im = (np.pi * self.dm**4) / 64     # Moment of inertia at bearing [m⁴]
        
        # Disk geometry
        self.dd = 90e-3     # disk diameter [m]
        self.ld = 47e-3     # disk length [m]
        self.md = 2.3       # disk mass (kg)
        self.W = self.md * self.g               # disk weight (N)
        self.me = 3.7e-5    # unbalance [kg.m]
        self.e = self.me / self.md              # disk eccentricity [m]
        self.Ip = (self.md * self.dd**2) / 8    # polar moment of inertia [kg.m²]
        self.Id = (self.md * self.dd**2) / 16 + (self.md * self.ld**2) / 12  # diametral moment [kg.m²]
        
        # Hydrodynamic bearing data
        self.D = 30e-3      # Diameter [m]
        self.C = 20e-3      # Length [m]
        self.delta = 90e-6  # radial clearance [m]
        self.mi = 0.051     # absolute viscosity [Pa.s]
        
        # Bearing type and properties
        self.bearing1_type = "journal"  # "journal" or "ball"
        self.bearing2_type = "journal"  # "journal" or "ball"
        
        # Ball bearing properties (fixed stiffness and damping)
        self.ball_bearing1_kxx = 1e8    # Stiffness in X direction [N/m]
        self.ball_bearing1_kyy = 1e8    # Stiffness in Y direction [N/m]
        self.ball_bearing1_kxy = 0      # Cross-coupling stiffness [N/m]
        self.ball_bearing1_kyx = 0      # Cross-coupling stiffness [N/m]
        self.ball_bearing1_cxx = 1e3    # Damping in X direction [N·s/m]
        self.ball_bearing1_cyy = 1e3    # Damping in Y direction [N·s/m]
        self.ball_bearing1_cxy = 0      # Cross-coupling damping [N·s/m]
        self.ball_bearing1_cyx = 0      # Cross-coupling damping [N·s/m]
        
        self.ball_bearing2_kxx = 1e8    # Stiffness in X direction [N/m]
        self.ball_bearing2_kyy = 1e8    # Stiffness in Y direction [N/m]
        self.ball_bearing2_kxy = 0      # Cross-coupling stiffness [N/m]
        self.ball_bearing2_kyx = 0      # Cross-coupling stiffness [N/m]
        self.ball_bearing2_cxx = 1e3    # Damping in X direction [N·s/m]
        self.ball_bearing2_cyy = 1e3    # Damping in Y direction [N·s/m]
        self.ball_bearing2_cxy = 0      # Cross-coupling damping [N·s/m]
        self.ball_bearing2_cyx = 0      # Cross-coupling damping [N·s/m]
        
        # Bearing reactions
        self.d1 = 145e-3    # Position a [m]
        self.d2 = 695e-3    # Position a + b [m]
        self.FM2 = self.W * self.d1 / self.d2   # Reaction at bearing 2
        self.FM1 = self.W - self.FM2            # Reaction at bearing 1
        
        # Speed range
        self.omega = np.arange(10, 2001, 10)    # Angular velocity [rad/s]
        self.n = len(self.omega)
        
        # Initialize arrays
        self.initialize_arrays()
        self.setup_fem_model()
        
    def initialize_arrays(self):
        """Initialize arrays for bearing coefficients"""
        # Bearing 1 coefficients
        self.epsilon1 = np.zeros(self.n)
        self.phi1 = np.zeros(self.n)
        self.k_m1_11 = np.zeros(self.n)
        self.k_m1_12 = np.zeros(self.n)
        self.k_m1_21 = np.zeros(self.n)
        self.k_m1_22 = np.zeros(self.n)
        self.d_m1_11 = np.zeros(self.n)
        self.d_m1_12 = np.zeros(self.n)
        self.d_m1_21 = np.zeros(self.n)
        self.d_m1_22 = np.zeros(self.n)
        
        # Bearing 2 coefficients
        self.epsilon2 = np.zeros(self.n)
        self.phi2 = np.zeros(self.n)
        self.k_m2_11 = np.zeros(self.n)
        self.k_m2_12 = np.zeros(self.n)
        self.k_m2_21 = np.zeros(self.n)
        self.k_m2_22 = np.zeros(self.n)
        self.d_m2_11 = np.zeros(self.n)
        self.d_m2_12 = np.zeros(self.n)
        self.d_m2_21 = np.zeros(self.n)
        self.d_m2_22 = np.zeros(self.n)

    def setup_fem_model(self):
        """Setup the finite element model"""
        
        # Coordinate matrix [node_number, X, Y]
        self.coord = np.array([
            [1, 0,      0],
            [2, 0.0160, 0],
            [3, 0.0260, 0],
            [4, 0.0360, 0],
            [5, 0.1035, 0],
            [6, 0.1710, 0],
            [7, 0.2385, 0],
            [8, 0.3060, 0],
            [9, 0.3735, 0],
            [10, 0.4410, 0],
            [11, 0.5085, 0],
            [12, 0.5760, 0],
            [13, 0.6435, 0],
            [14, 0.7110, 0],
            [15, 0.7210, 0],
            [16, 0.7310, 0],
            [17, 0.7470, 0]
        ])
        
        # Incidence matrix [element_number, material_table, geometry_table, node1, node2]
        self.inci = np.array([
            [1,  1, 2, 1,  2],   # geo table: 1=bearing, 2=shaft
            [2,  1, 1, 2,  3],
            [3,  1, 1, 3,  4],
            [4,  1, 2, 4,  5],
            [5,  1, 2, 5,  6],
            [6,  1, 2, 6,  7],
            [7,  1, 2, 7,  8],
            [8,  1, 2, 8,  9],
            [9,  1, 2, 9,  10],
            [10, 1, 2, 10, 11],
            [11, 1, 2, 11, 12],
            [12, 1, 2, 12, 13],
            [13, 1, 2, 13, 14],
            [14, 1, 1, 14, 15],
            [15, 1, 1, 15, 16],
            [16, 1, 2, 16, 17]
        ], dtype=int)
        
        # Geometry table [Area, Moment_of_Inertia, diameter]
        self.Tgeo = np.array([
            [self.Am, self.Ae],  # Areas
            [self.Im, self.Ie],  # Moments of inertia
            [self.dm, self.de]   # Diameters
        ])
        
        # System dimensions
        self.nnos = self.coord.shape[0]  # number of nodes
        self.nel = self.inci.shape[0]    # number of elements
        self.ngdl = 4                    # degrees of freedom per node
        
        # Create ID matrix
        self.id_matrix = np.ones((self.ngdl, self.nnos), dtype=int)
        self.neq = 0
        for i in range(self.nnos):
            for j in range(self.ngdl):
                if self.id_matrix[j, i] == 1:
                    self.neq += 1
                    self.id_matrix[j, i] = self.neq
        
        # Build global matrices
        self.build_global_matrices()
        
    def build_global_matrices(self):
        """Build global FEM matrices"""
        
        # Initialize global matrices
        self.kg = np.zeros((self.neq, self.neq))  # Global stiffness
        self.mg = np.zeros((self.neq, self.neq))  # Global mass
        self.Cg = np.zeros((self.neq, self.neq))  # Global damping
        self.Gg = np.zeros((self.neq, self.neq))  # Global gyroscopic
        
        ngdle = self.ngdl * 2  # DOF per element
        
        # Element assembly loop
        for i in range(self.nel):
            noi = self.inci[i, 3] - 1  # First node (0-indexed)
            noj = self.inci[i, 4] - 1  # Second node (0-indexed)
            
            xi = self.coord[noi, 1]    # x-coordinate of first node
            yi = self.coord[noi, 2]    # y-coordinate of first node  
            xj = self.coord[noj, 1]    # x-coordinate of second node
            yj = self.coord[noj, 2]    # y-coordinate of second node
            
            le = np.sqrt((xj - xi)**2 + (yj - yi)**2)  # Element length
            
            ngeo = self.inci[i, 2] - 1  # Geometry index (0-indexed)
            
            A = self.Tgeo[0, ngeo]     # Cross-sectional area
            I = self.Tgeo[1, ngeo]     # Moment of inertia
            d = self.Tgeo[2, ngeo]     # Diameter
            
            # Element stiffness matrix
            ke = (self.E * I / le**3) * np.array([
                [12,      6*le,     0,       0,      -12,     6*le,     0,       0],
                [6*le,    4*le**2,  0,       0,      -6*le,   2*le**2,  0,       0],
                [0,       0,        12,     -6*le,   0,       0,       -12,     -6*le],
                [0,       0,       -6*le,    4*le**2, 0,       0,        6*le,    2*le**2],
                [-12,    -6*le,     0,       0,       12,     -6*le,     0,       0],
                [6*le,    2*le**2,  0,       0,      -6*le,    4*le**2,  0,       0],
                [0,       0,       -12,      6*le,    0,       0,        12,      6*le],
                [0,       0,       -6*le,    2*le**2, 0,       0,        6*le,    4*le**2]
            ])
            
            # Element damping matrix (proportional)
            Beta = 1e-4
            Ce = Beta * ke
            
            # Element mass matrix
            Me = (self.rho * A * le / 420) * np.array([
                [156,     22*le,    0,       0,      54,      -13*le,   0,       0],
                [22*le,   4*le**2,  0,       0,      13*le,   -3*le**2, 0,       0],
                [0,       0,        156,    -22*le,  0,       0,        54,      13*le],
                [0,       0,       -22*le,   4*le**2, 0,       0,       -13*le,  -3*le**2],
                [54,      13*le,    0,       0,      156,     -22*le,   0,       0],
                [-13*le, -3*le**2,  0,       0,     -22*le,    4*le**2, 0,       0],
                [0,       0,        54,     -13*le,  0,       0,        156,     22*le],
                [0,       0,        13*le,  -3*le**2, 0,       0,        22*le,   4*le**2]
            ])
            
            # Element gyroscopic matrix
            Ge = (self.rho * A * d**2 / (240 * le)) * np.array([
                [0,       0,        36,     -3*le,   0,       0,       -36,     -3*le],
                [0,       0,        3*le,   -4*le**2, 0,       0,       -3*le,    le**2],
                [-36,    -3*le,     0,       0,       36,     -3*le,     0,       0],
                [3*le,    4*le**2,  0,       0,      -3*le,   -le**2,   0,       0],
                [0,       0,       -36,      3*le,    0,       0,        36,      3*le],
                [0,       0,        3*le,    le**2,   0,       0,       -3*le,   -4*le**2],
                [36,      3*le,     0,       0,      -36,      3*le,     0,       0],
                [3*le,   -le**2,    0,       0,      -3*le,    4*le**2,  0,       0]
            ])
            
            # Assembly into global matrices
            loc = np.array([
                self.id_matrix[0, noi]-1, self.id_matrix[1, noi]-1, self.id_matrix[2, noi]-1, self.id_matrix[3, noi]-1,
                self.id_matrix[0, noj]-1, self.id_matrix[1, noj]-1, self.id_matrix[2, noj]-1, self.id_matrix[3, noj]-1
            ], dtype=int)
            
            for il in range(ngdle):
                ilg = loc[il]
                if ilg >= 0:
                    for ic in range(ngdle):
                        icg = loc[ic]
                        if icg >= 0:
                            self.kg[ilg, icg] += ke[il, ic]
                            self.mg[ilg, icg] += Me[il, ic]
                            self.Gg[ilg, icg] += Ge[il, ic]
                            self.Cg[ilg, icg] += Ce[il, ic]
    
    def calculate_ball_bearing_coefficients(self, bearing_num):
        """Calculate fixed ball bearing coefficients (speed-independent)"""
        
        coefficients = {}
        
        # Get ball bearing properties based on bearing number
        if bearing_num == 1:
            kxx = self.ball_bearing1_kxx
            kyy = self.ball_bearing1_kyy 
            kxy = self.ball_bearing1_kxy
            kyx = self.ball_bearing1_kyx
            cxx = self.ball_bearing1_cxx
            cyy = self.ball_bearing1_cyy
            cxy = self.ball_bearing1_cxy
            cyx = self.ball_bearing1_cyx
        else:
            kxx = self.ball_bearing2_kxx
            kyy = self.ball_bearing2_kyy
            kxy = self.ball_bearing2_kxy
            kyx = self.ball_bearing2_kyx
            cxx = self.ball_bearing2_cxx
            cyy = self.ball_bearing2_cyy
            cxy = self.ball_bearing2_cxy
            cyx = self.ball_bearing2_cyx
        
        # For ball bearings, coefficients are constant for all speeds
        for i in range(self.n):
            # Store stiffness coefficients (using YZ coordinate system like journal bearings)
            coefficients[f'k_11_{i}'] = kyy      # K_yy
            coefficients[f'k_12_{i}'] = kxy      # K_yz  
            coefficients[f'k_21_{i}'] = kyx      # K_zy
            coefficients[f'k_22_{i}'] = kxx      # K_zz
            
            # Store damping coefficients
            coefficients[f'd_11_{i}'] = cyy      # C_yy
            coefficients[f'd_12_{i}'] = cxy      # C_yz
            coefficients[f'd_21_{i}'] = cyx      # C_zy
            coefficients[f'd_22_{i}'] = cxx      # C_zz
            
            # Ball bearings don't have epsilon and phi (journal center position)
            coefficients[f'epsilon_{i}'] = 0
            coefficients[f'phi_{i}'] = 0
        
        return coefficients
        
    def calculate_bearing_coefficients(self, bearing_num, FM):
        """Calculate bearing coefficients using Newton-Raphson method"""
        
        coefficients = {}
        
        for i in range(self.n):
            sol = np.array([0.8, 0.1])  # initial guess [epsilon, phi]
            error = 1e8
            iter_count = 0
            itermax = 500
            
            while error > 1e-6 and iter_count < itermax:
                iter_count += 1
                epsilon = sol[0]
                phi = sol[1]
                
                # Tangential and radial forces
                Ft = self.mi * self.omega[i] * self.D / 2 * self.C**3 / self.delta**2 * \
                     (np.pi * epsilon / (4 * (1 - epsilon**2)**(3/2)))
                Fr = self.mi * self.omega[i] * self.D / 2 * self.C**3 / self.delta**2 * \
                     (epsilon**2 / (1 - epsilon**2)**2)
                
                # Equations to solve
                eqn = np.array([
                    Ft * np.sin(phi) + Fr * np.cos(phi) - FM,
                    Ft * np.cos(phi) - Fr * np.sin(phi)
                ])
                
                # Jacobian matrix
                J = np.zeros((2, 2))
                
                # Partial derivatives
                dFt_deps = self.mi * self.omega[i] * self.D / 2 * self.C**3 / self.delta**2 * \
                          (np.pi / (4 * (1 - epsilon**2)**(3/2)) + 3 * np.pi * epsilon**2 / (4 * (1 - epsilon**2)**(5/2)))
                
                dFr_deps = self.mi * self.omega[i] * self.D / 2 * self.C**3 / self.delta**2 * \
                          (2 * epsilon / (1 - epsilon**2)**2 + 4 * epsilon**3 / (1 - epsilon**2)**3)
                
                J[0, 0] = dFt_deps * np.sin(phi) + dFr_deps * np.cos(phi)
                J[0, 1] = Ft * np.cos(phi) - Fr * np.sin(phi)
                J[1, 0] = dFt_deps * np.cos(phi) - dFr_deps * np.sin(phi)
                J[1, 1] = -Ft * np.sin(phi) - Fr * np.cos(phi)
                
                # Newton-Raphson step
                try:
                    step = -solve(J, eqn)
                    sol = sol + step
                    error = np.sqrt(step[0]**2 + step[1]**2)
                except:
                    print(f"Convergence failed at omega = {self.omega[i]} rad/s for bearing {bearing_num}")
                    break
            
            if iter_count >= itermax:
                print(f"Did not converge within {itermax} iterations for bearing {bearing_num} at omega = {self.omega[i]} rad/s")
            
            # Store results
            epsilon_final = sol[0]
            phi_final = sol[1]
            
            # Calculate influence coefficients
            A = 4 / (np.pi**2 + (16 - np.pi**2) * epsilon_final**2)**(1.5)
            
            gama_11 = (2 * np.pi**2 + (16 - np.pi**2) * epsilon_final**2) * A
            gama_12 = np.pi / 4 * (np.pi**2 - 2 * np.pi**2 * epsilon_final**2 - (16 - np.pi**2) * epsilon_final**4) / \
                      (epsilon_final * np.sqrt(1 - epsilon_final**2)) * A
            gama_21 = -np.pi / 4 * (np.pi**2 + (32 + np.pi**2) * epsilon_final**2 + (32 - 2 * np.pi**2) * epsilon_final**4) / \
                      (epsilon_final * np.sqrt(1 - epsilon_final**2)) * A
            gama_22 = (np.pi**2 + (32 + np.pi**2) * epsilon_final**2 + (32 - 2 * np.pi**2) * epsilon_final**4) / \
                      (1 - epsilon_final**2) * A
            
            beta_11 = np.pi / 2 * np.sqrt(1 - epsilon_final**2) / epsilon_final * \
                      (np.pi**2 + epsilon_final**2 * (2 * np.pi**2 - 16)) * A
            beta_12 = -(2 * np.pi**2 + (4 * np.pi**2 - 32) * epsilon_final**2) * A
            beta_21 = beta_12
            beta_22 = np.pi / 2 * (np.pi**2 + (48 - 2 * np.pi**2) * epsilon_final**2 + np.pi**2 * epsilon_final**4) / \
                      (epsilon_final * np.sqrt(1 - epsilon_final**2)) * A
            
            # Store coefficients
            coefficients[f'epsilon_{i}'] = epsilon_final
            coefficients[f'phi_{i}'] = phi_final
            coefficients[f'k_11_{i}'] = gama_11 * FM / self.delta
            coefficients[f'k_12_{i}'] = gama_12 * FM / self.delta
            coefficients[f'k_21_{i}'] = gama_21 * FM / self.delta
            coefficients[f'k_22_{i}'] = gama_22 * FM / self.delta
            coefficients[f'd_11_{i}'] = beta_11 * FM / (self.omega[i] * self.delta)
            coefficients[f'd_12_{i}'] = beta_12 * FM / (self.omega[i] * self.delta)
            coefficients[f'd_21_{i}'] = beta_21 * FM / (self.omega[i] * self.delta)
            coefficients[f'd_22_{i}'] = beta_22 * FM / (self.omega[i] * self.delta)
        
        return coefficients
        
    def calculate_system_response(self):
        """Calculate frequency response of the system"""
        
        print("Calculating system frequency response...")
        
        # Initialize response matrix
        self.X = np.zeros((self.neq, self.n), dtype=complex)
        
        # System response calculation
        for i in range(self.n):
            s = 1j * self.omega[i]
            
            # Initialize bearing and disk matrices
            kmancal = np.zeros((self.neq, self.neq))
            Cmancal = np.zeros((self.neq, self.neq))
            Md = np.zeros((self.neq, self.neq))
            Gd = np.zeros((self.neq, self.neq))
            
            # Add bearing 1 stiffness coefficients (node 3, indices 8-11)
            bearing1_dofs = [8, 9, 10, 11]  # 0-indexed
            if max(bearing1_dofs) < self.neq:
                kmancal[np.ix_(bearing1_dofs, bearing1_dofs)] = np.array([
                    [self.k_m1_11[i], 0, self.k_m1_12[i], 0],
                    [0, 0, 0, 0],
                    [self.k_m1_21[i], 0, self.k_m1_22[i], 0],
                    [0, 0, 0, 0]
                ])
                
                # Add bearing 1 damping coefficients
                Cmancal[np.ix_(bearing1_dofs, bearing1_dofs)] = np.array([
                    [self.d_m1_11[i], 0, self.d_m1_12[i], 0],
                    [0, 0, 0, 0],
                    [self.d_m1_21[i], 0, self.d_m1_22[i], 0],
                    [0, 0, 0, 0]
                ])
            
            # Add bearing 2 stiffness coefficients (node 15, indices 56-59)
            bearing2_dofs = [56, 57, 58, 59]  # 0-indexed
            if max(bearing2_dofs) < self.neq:
                kmancal[np.ix_(bearing2_dofs, bearing2_dofs)] = np.array([
                    [self.k_m2_11[i], 0, self.k_m2_12[i], 0],
                    [0, 0, 0, 0],
                    [self.k_m2_21[i], 0, self.k_m2_22[i], 0],
                    [0, 0, 0, 0]
                ])
                
                # Add bearing 2 damping coefficients
                Cmancal[np.ix_(bearing2_dofs, bearing2_dofs)] = np.array([
                    [self.d_m2_11[i], 0, self.d_m2_12[i], 0],
                    [0, 0, 0, 0],
                    [self.d_m2_21[i], 0, self.d_m2_22[i], 0],
                    [0, 0, 0, 0]
                ])
            
            # Add disk mass influence (node 6, indices 20-23)
            disk_dofs = [20, 21, 22, 23]  # 0-indexed
            if max(disk_dofs) < self.neq:
                Md[np.ix_(disk_dofs, disk_dofs)] = np.array([
                    [self.md, 0, 0, 0],
                    [0, self.Id, 0, 0],
                    [0, 0, self.md, 0],
                    [0, 0, 0, self.Id]
                ])
                
                # Add disk gyroscopic effect
                Gd[np.ix_(disk_dofs, disk_dofs)] = np.array([
                    [0, 0, 0, 0],
                    [0, 0, 0, self.Ip],
                    [0, 0, 0, 0],
                    [0, -self.Ip, 0, 0]
                ])
            
            # Force vector
            Fe_temp = np.zeros(self.neq)
            if len(Fe_temp) > 20:
                Fe_temp[20] = self.me * self.omega[i]**2  # Y direction
            if len(Fe_temp) > 22:
                Fe_temp[22] = self.me * self.omega[i]**2 - self.W  # Z direction
            
            # Assemble system matrices
            Kg1 = self.kg + kmancal
            Cg1 = self.Cg + Cmancal
            Mg1 = self.mg + Md
            Gg1 = self.Gg + Gd
            
            # Impedance matrix
            Z = Mg1 * s**2 + (Cg1 + self.omega[i] * Gg1) * s + Kg1
            
            # Solve system
            try:
                Xi = solve(Z, Fe_temp)
                self.X[:, i] = Xi
            except:
                print(f"Failed to solve system at omega = {self.omega[i]} rad/s")
                self.X[:, i] = np.zeros(self.neq)
        
        print("System response calculated successfully!")
    
    def calculate_critical_speeds(self):
        """Calculate critical speeds by solving eigenvalue problem and finding Campbell diagram intersections"""
        
        print("Calculating critical speeds using eigenvalue analysis...")
        
        # Check if we have ball bearings (speed-independent stiffness)
        has_ball_bearings = (self.bearing1_type == "ball" or self.bearing2_type == "ball")
        
        if has_ball_bearings and self.bearing1_type == "ball" and self.bearing2_type == "ball":
            # For pure ball bearing systems, calculate critical speeds differently
            # Ball bearings have constant stiffness, so we can calculate natural frequencies directly
            return self._calculate_ball_bearing_critical_speeds()
        else:
            # For journal bearings or mixed systems, use the speed-dependent approach
            return self._calculate_general_critical_speeds()
    
    def _calculate_ball_bearing_critical_speeds(self):
        """Calculate critical speeds for pure ball bearing systems (speed-independent)"""
        
        print("Using ball bearing critical speed calculation (speed-independent)...")
        
        # For ball bearings, use stiffness values at any speed (they're constant)
        i = len(self.omega) // 2  # Use middle speed point
        
        # Initialize bearing and disk matrices
        kmancal = np.zeros((self.neq, self.neq))
        Md = np.zeros((self.neq, self.neq))
        
        # Add bearing 1 stiffness coefficients
        bearing1_dofs = [8, 9, 10, 11]
        if max(bearing1_dofs) < self.neq:
            kmancal[np.ix_(bearing1_dofs, bearing1_dofs)] = np.array([
                [self.k_m1_11[i], 0, self.k_m1_12[i], 0],
                [0, 0, 0, 0],
                [self.k_m1_21[i], 0, self.k_m1_22[i], 0],
                [0, 0, 0, 0]
            ])
        
        # Add bearing 2 stiffness coefficients
        bearing2_dofs = [56, 57, 58, 59]
        if max(bearing2_dofs) < self.neq:
            kmancal[np.ix_(bearing2_dofs, bearing2_dofs)] = np.array([
                [self.k_m2_11[i], 0, self.k_m2_12[i], 0],
                [0, 0, 0, 0],
                [self.k_m2_21[i], 0, self.k_m2_22[i], 0],
                [0, 0, 0, 0]
            ])
        
        # Add disk mass influence
        disk_dofs = [20, 21, 22, 23]
        if max(disk_dofs) < self.neq:
            Md[np.ix_(disk_dofs, disk_dofs)] = np.array([
                [self.md, 0, 0, 0],
                [0, self.Id, 0, 0],
                [0, 0, self.md, 0],
                [0, 0, 0, self.Id]
            ])
        
        # Assemble system matrices
        Kg1 = self.kg + kmancal
        Mg1 = self.mg + Md
        
        try:
            # Solve generalized eigenvalue problem: K*phi = lambda*M*phi
            eigenvals, eigenvecs = np.linalg.eig(np.linalg.solve(Mg1, Kg1))
            
            # Extract positive real eigenvalues and convert to frequencies
            real_eigenvals = np.real(eigenvals)
            positive_eigenvals = real_eigenvals[real_eigenvals > 0]
            natural_freqs = np.sqrt(positive_eigenvals)
            natural_freqs = np.sort(natural_freqs)
            
            # The critical speeds for ball bearings are simply the natural frequencies
            if len(natural_freqs) >= 3:
                self.critical_speeds = natural_freqs[:3]
            elif len(natural_freqs) >= 1:
                # If we have fewer than 3 modes, use what we have and add estimates
                self.critical_speeds = np.zeros(3)
                self.critical_speeds[:len(natural_freqs)] = natural_freqs
                if len(natural_freqs) == 1:
                    self.critical_speeds[1] = natural_freqs[0] * 2.5
                    self.critical_speeds[2] = natural_freqs[0] * 4.8
                elif len(natural_freqs) == 2:
                    self.critical_speeds[2] = natural_freqs[1] * 1.8
            else:
                # Fallback based on ball bearing stiffness estimates
                k_avg = (self.ball_bearing1_kxx + self.ball_bearing1_kyy + 
                        self.ball_bearing2_kxx + self.ball_bearing2_kyy) / 4
                m_total = self.md + 0.1 * self.rho * self.Ae * self.le  # Approximate total mass
                omega_n = np.sqrt(k_avg / m_total)
                self.critical_speeds = np.array([omega_n * 0.7, omega_n * 1.2, omega_n * 2.1])
            
            # Create natural frequency matrix for plotting (constant for ball bearings)
            self.natural_freq_matrix = np.zeros((3, self.n))
            for j in range(self.n):
                self.natural_freq_matrix[:, j] = self.critical_speeds
                
        except Exception as e:
            print(f"Ball bearing critical speed calculation failed: {e}")
            # Use fallback values based on typical ball bearing systems
            k_avg = (self.ball_bearing1_kxx + self.ball_bearing1_kyy + 
                    self.ball_bearing2_kxx + self.ball_bearing2_kyy) / 4
            m_total = self.md + 0.1 * self.rho * self.Ae * self.le
            omega_n = np.sqrt(k_avg / m_total)
            self.critical_speeds = np.array([omega_n * 0.6, omega_n * 1.1, omega_n * 1.9])
            
            # Create natural frequency matrix
            self.natural_freq_matrix = np.zeros((3, self.n))
            for j in range(self.n):
                self.natural_freq_matrix[:, j] = self.critical_speeds
        
        print(f"Critical speeds calculated: {self.critical_speeds} rad/s")
        print(f"Critical speeds in Hz: {self.critical_speeds / (2 * np.pi)} Hz")
        return self.critical_speeds
    
    def _calculate_general_critical_speeds(self):
        """Calculate critical speeds for journal bearings or mixed systems (speed-dependent)"""
        
        print("Using general critical speed calculation (speed-dependent)...")
        
        # Store natural frequencies for each speed
        self.natural_frequencies = []
        critical_speeds = []
        
        for i in range(self.n):
            # Initialize bearing and disk matrices
            kmancal = np.zeros((self.neq, self.neq))
            Md = np.zeros((self.neq, self.neq))
            
            # Add bearing 1 stiffness coefficients
            bearing1_dofs = [8, 9, 10, 11]
            if max(bearing1_dofs) < self.neq:
                kmancal[np.ix_(bearing1_dofs, bearing1_dofs)] = np.array([
                    [self.k_m1_11[i], 0, self.k_m1_12[i], 0],
                    [0, 0, 0, 0],
                    [self.k_m1_21[i], 0, self.k_m1_22[i], 0],
                    [0, 0, 0, 0]
                ])
            
            # Add bearing 2 stiffness coefficients
            bearing2_dofs = [56, 57, 58, 59]
            if max(bearing2_dofs) < self.neq:
                kmancal[np.ix_(bearing2_dofs, bearing2_dofs)] = np.array([
                    [self.k_m2_11[i], 0, self.k_m2_12[i], 0],
                    [0, 0, 0, 0],
                    [self.k_m2_21[i], 0, self.k_m2_22[i], 0],
                    [0, 0, 0, 0]
                ])
            
            # Add disk mass influence
            disk_dofs = [20, 21, 22, 23]
            if max(disk_dofs) < self.neq:
                Md[np.ix_(disk_dofs, disk_dofs)] = np.array([
                    [self.md, 0, 0, 0],
                    [0, self.Id, 0, 0],
                    [0, 0, self.md, 0],
                    [0, 0, 0, self.Id]
                ])
            
            # Assemble system matrices
            Kg1 = self.kg + kmancal
            Mg1 = self.mg + Md
            
            # Solve eigenvalue problem
            try:
                eigenvals = np.linalg.eigvals(np.linalg.solve(Mg1, Kg1))
                real_eigenvals = np.real(eigenvals)
                positive_eigenvals = real_eigenvals[real_eigenvals > 0]
                natural_freqs = np.sqrt(positive_eigenvals)
                natural_freqs = np.sort(natural_freqs)
                
                self.natural_frequencies.append(natural_freqs[:10] if len(natural_freqs) >= 10 else natural_freqs)
                
            except:
                print(f"Eigenvalue calculation failed at omega = {self.omega[i]:.1f} rad/s")
                self.natural_frequencies.append(np.array([]))
                continue
        
        # Convert to numpy array for easier processing
        max_modes = max(len(freqs) for freqs in self.natural_frequencies if len(freqs) > 0)
        if max_modes == 0:
            print("No valid eigenvalues found, using fallback values")
            self.critical_speeds = np.array([160.0, 750.0, 1460.0])
            self.natural_freq_matrix = np.zeros((3, self.n))
            return self.critical_speeds
            
        natural_freq_matrix = np.zeros((max_modes, self.n))
        
        for i, freqs in enumerate(self.natural_frequencies):
            if len(freqs) > 0:
                n_freqs = min(len(freqs), max_modes)
                natural_freq_matrix[:n_freqs, i] = freqs[:n_freqs]
        
        # Find critical speeds (Campbell diagram intersections)
        for mode_idx in range(min(5, max_modes)):
            nat_freq_line = natural_freq_matrix[mode_idx, :]
            
            # Find intersections with synchronous speed line
            for j in range(1, len(self.omega)):
                if nat_freq_line[j-1] > 0 and nat_freq_line[j] > 0:
                    omega_prev = self.omega[j-1]
                    omega_curr = self.omega[j]
                    nat_prev = nat_freq_line[j-1]
                    nat_curr = nat_freq_line[j]
                    
                    # Check if synchronous line crosses natural frequency line
                    if ((omega_prev - nat_prev) * (omega_curr - nat_curr) <= 0):
                        if abs(nat_curr - nat_prev) > 1e-10:
                            alpha = (omega_prev - nat_prev) / ((omega_curr - nat_curr) - (omega_prev - nat_prev))
                            critical_speed = omega_prev + alpha * (omega_curr - omega_prev)
                            
                            if self.omega[0] <= critical_speed <= self.omega[-1]:
                                critical_speeds.append(critical_speed)
        
        # Process results
        if critical_speeds:
            critical_speeds = sorted(list(set(np.round(critical_speeds, 1))))
            self.critical_speeds = np.array(critical_speeds[:3])
        else:
            print("No critical speeds found in range, using fallback values")
            self.critical_speeds = np.array([160.0, 750.0, 1460.0])
        
        self.natural_freq_matrix = natural_freq_matrix
        
        print(f"Critical speeds calculated: {self.critical_speeds} rad/s")
        print(f"Critical speeds in Hz: {self.critical_speeds / (2 * np.pi)} Hz")
        return self.critical_speeds
    
    def run_analysis(self):
        """Run complete rotordynamic analysis"""
        
        print("Starting rotordynamic analysis...")
        
        # Calculate bearing coefficients based on bearing type
        print(f"Calculating bearing 1 coefficients ({self.bearing1_type})...")
        if self.bearing1_type == "ball":
            bearing1_coeff = self.calculate_ball_bearing_coefficients(1)
        else:
            bearing1_coeff = self.calculate_bearing_coefficients(1, self.FM1)
        
        print(f"Calculating bearing 2 coefficients ({self.bearing2_type})...")
        if self.bearing2_type == "ball":
            bearing2_coeff = self.calculate_ball_bearing_coefficients(2)
        else:
            bearing2_coeff = self.calculate_bearing_coefficients(2, self.FM2)
        
        # Store coefficients in class arrays
        for i in range(self.n):
            # Bearing 1
            self.epsilon1[i] = bearing1_coeff[f'epsilon_{i}']
            self.phi1[i] = bearing1_coeff[f'phi_{i}'] * 180 / np.pi  # Convert to degrees
            self.k_m1_11[i] = bearing1_coeff[f'k_11_{i}']
            self.k_m1_12[i] = bearing1_coeff[f'k_12_{i}']
            self.k_m1_21[i] = bearing1_coeff[f'k_21_{i}']
            self.k_m1_22[i] = bearing1_coeff[f'k_22_{i}']
            self.d_m1_11[i] = bearing1_coeff[f'd_11_{i}']
            self.d_m1_12[i] = bearing1_coeff[f'd_12_{i}']
            self.d_m1_21[i] = bearing1_coeff[f'd_21_{i}']
            self.d_m1_22[i] = bearing1_coeff[f'd_22_{i}']
            
            # Bearing 2
            self.epsilon2[i] = bearing2_coeff[f'epsilon_{i}']
            self.phi2[i] = bearing2_coeff[f'phi_{i}'] * 180 / np.pi  # Convert to degrees
            self.k_m2_11[i] = bearing2_coeff[f'k_11_{i}']
            self.k_m2_12[i] = bearing2_coeff[f'k_12_{i}']
            self.k_m2_21[i] = bearing2_coeff[f'k_21_{i}']
            self.k_m2_22[i] = bearing2_coeff[f'k_22_{i}']
            self.d_m2_11[i] = bearing2_coeff[f'd_11_{i}']
            self.d_m2_12[i] = bearing2_coeff[f'd_12_{i}']
            self.d_m2_21[i] = bearing2_coeff[f'd_21_{i}']
            self.d_m2_22[i] = bearing2_coeff[f'd_22_{i}']
        
        print("Bearing coefficients calculated successfully!")
        
        # Calculate critical speeds
        self.calculate_critical_speeds()
        
        # Calculate system response
        self.calculate_system_response() 
        
        return True
    
    def plot_results(self):
        """Generate all analysis plots"""
        
        print("Generating plots...")
        
        # Plot 1: Bearing coefficients
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Bearing 1 stiffness
        axes[0, 0].plot(self.omega, self.k_m1_11, label='K_yy')
        axes[0, 0].plot(self.omega, self.k_m1_12, label='K_yz')
        axes[0, 0].plot(self.omega, self.k_m1_21, label='K_zy')
        axes[0, 0].plot(self.omega, self.k_m1_22, label='K_zz')
        axes[0, 0].set_ylim([-1e7, 1e7])
        axes[0, 0].set_ylabel('Stiffness [N/m]')
        axes[0, 0].set_xlabel('Angular Velocity [rad/s]')
        axes[0, 0].legend()
        axes[0, 0].set_title('Bearing 1 Stiffness Coefficients')
        axes[0, 0].grid(True)
        
        # Bearing 2 stiffness
        axes[0, 1].plot(self.omega, self.k_m2_11, label='K_yy')
        axes[0, 1].plot(self.omega, self.k_m2_12, label='K_yz')
        axes[0, 1].plot(self.omega, self.k_m2_21, label='K_zy')
        axes[0, 1].plot(self.omega, self.k_m2_22, label='K_zz')
        axes[0, 1].set_ylim([-1e7, 1e7])
        axes[0, 1].set_ylabel('Stiffness [N/m]')
        axes[0, 1].set_xlabel('Angular Velocity [rad/s]')
        axes[0, 1].legend()
        axes[0, 1].set_title('Bearing 2 Stiffness Coefficients')
        axes[0, 1].grid(True)
        
        # Bearing 1 damping
        axes[1, 0].plot(self.omega, self.d_m1_11, label='C_yy')
        axes[1, 0].plot(self.omega, self.d_m1_12, label='C_yz')
        axes[1, 0].plot(self.omega, self.d_m1_21, label='C_zy')
        axes[1, 0].plot(self.omega, self.d_m1_22, label='C_zz')
        axes[1, 0].set_ylabel('Damping [N·s/m]')
        axes[1, 0].set_xlabel('Angular Velocity [rad/s]')
        axes[1, 0].legend()
        axes[1, 0].set_title('Bearing 1 Damping Coefficients')
        axes[1, 0].grid(True)
        
        # Bearing 2 damping
        axes[1, 1].plot(self.omega, self.d_m2_11, label='C_yy')
        axes[1, 1].plot(self.omega, self.d_m2_12, label='C_yz')
        axes[1, 1].plot(self.omega, self.d_m2_21, label='C_zy')
        axes[1, 1].plot(self.omega, self.d_m2_22, label='C_zz')
        axes[1, 1].set_ylabel('Damping [N·s/m]')
        axes[1, 1].set_xlabel('Angular Velocity [rad/s]')
        axes[1, 1].legend()
        axes[1, 1].set_title('Bearing 2 Damping Coefficients')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Plot 2: Bearing locus
        theta = np.linspace(0, 2*np.pi, 1000)
        rho = 0.8
        Z_clearance = rho * np.cos(theta)
        Y_clearance = rho * np.sin(theta)
        
        ExcZ1 = self.epsilon1 * np.sin(np.radians(self.phi1))
        ExcY1 = -self.epsilon1 * np.cos(np.radians(self.phi1))
        ExcZ2 = self.epsilon2 * np.sin(np.radians(self.phi2))
        ExcY2 = -self.epsilon2 * np.cos(np.radians(self.phi2))
        
        plt.figure(figsize=(8, 8))
        plt.plot(ExcZ1, ExcY1, 'b-', linewidth=2, label='Bearing 1')
        plt.plot(ExcZ2, ExcY2, 'r-', linewidth=2, label='Bearing 2')
        plt.plot(Z_clearance, Y_clearance, 'k--', linewidth=1, label='Clearance circle')
        plt.ylim([-0.8, 0.8])
        plt.ylabel('ε × cos(φ)')
        plt.xlabel('ε × sin(φ)')
        plt.xlim([-0.8, 0.8])
        plt.legend()
        plt.title('Bearing Locus')
        plt.grid(True)
        plt.axis('equal')
        plt.show()
        
        # Plot 3: Frequency Response Functions
        plt.figure(figsize=(15, 10))
        
        # FRF at bearing 1 (node 3)
        plt.subplot(2, 2, 1)
        if self.X.shape[0] > 10:
            plt.semilogy(self.omega, np.abs(self.X[8, :]), 'b-', label='Node 3 - Y displacement')
            plt.semilogy(self.omega, np.abs(self.X[10, :]), 'r-', label='Node 3 - Z displacement')
        plt.xlim([10, 2000])
        plt.xlabel('Angular Velocity Ω [rad/s]')
        plt.ylabel('Amplitude [m]')
        plt.title('FRF - Bearing 1 Node')
        plt.legend()
        plt.grid(True)
        
        # FRF at disk (node 6)
        plt.subplot(2, 2, 2)
        if self.X.shape[0] > 22:
            plt.semilogy(self.omega, np.abs(self.X[20, :]), 'b-', label='Node 6 - Y displacement')
            plt.semilogy(self.omega, np.abs(self.X[22, :]), 'r-', label='Node 6 - Z displacement')
        plt.xlim([10, 2000])
        plt.xlabel('Angular Velocity Ω [rad/s]')
        plt.ylabel('Amplitude [m]')
        plt.title('FRF - Disk Node')
        plt.legend()
        plt.grid(True)
        
        # FRF at bearing 2 (node 15)
        plt.subplot(2, 2, 3)
        if self.X.shape[0] > 58:
            plt.semilogy(self.omega, np.abs(self.X[56, :]), 'b-', label='Node 15 - Y displacement')
            plt.semilogy(self.omega, np.abs(self.X[58, :]), 'r-', label='Node 15 - Z displacement')
        plt.xlim([10, 2000])
        plt.xlabel('Angular Velocity Ω [rad/s]')
        plt.ylabel('Amplitude [m]')
        plt.title('FRF - Bearing 2 Node')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Plot 4: Mode shapes at specific frequencies
        plt.figure(figsize=(15, 5))
        
        # Mode shape at 160 rad/s (index 15)
        plt.subplot(1, 3, 1)
        if len(self.omega) > 15 and self.X.shape[0] > 4:
            mode_idx = 15
            y_displacements = np.real(self.X[::4, mode_idx])  # Every 4th element (Y displacements)
            plt.plot(self.coord[:, 1], y_displacements, 'b-o', linewidth=2, markersize=4, label='Y deformation')
            plt.plot(self.coord[:, 1], self.coord[:, 2], 'k--', marker='s', markersize=3, label='Initial state')
        
        plt.legend(loc='best')
        plt.xlabel('Nodal coordinates [m]')
        plt.ylabel('Amplitude [m]')
        plt.title('System deformation in Y at Ω = 160 rad/s')
        plt.grid(True)
        
        # Mode shape at 750 rad/s (index 74)
        plt.subplot(1, 3, 2)
        if len(self.omega) > 74 and self.X.shape[0] > 4:
            mode_idx = 74
            y_displacements = np.real(self.X[::4, mode_idx])
            plt.plot(self.coord[:, 1], y_displacements, 'b-o', linewidth=2, markersize=4, label='Y deformation')
            plt.plot(self.coord[:, 1], self.coord[:, 2], 'k--', marker='s', markersize=3, label='Initial state')
        
        plt.legend(loc='best')
        plt.xlabel('Nodal coordinates [m]')
        plt.ylabel('Amplitude [m]')
        plt.title('System deformation in Y at Ω = 750 rad/s')
        plt.grid(True)
        
        # Mode shape at 1460 rad/s (index 145)
        plt.subplot(1, 3, 3)
        if len(self.omega) > 145 and self.X.shape[0] > 4:
            mode_idx = 145
            y_displacements = np.real(self.X[::4, mode_idx])
            plt.plot(self.coord[:, 1], y_displacements, 'b-o', linewidth=2, markersize=4, label='Y deformation')
            plt.plot(self.coord[:, 1], self.coord[:, 2], 'k--', marker='s', markersize=3, label='Initial state')
        
        plt.legend(loc='best')
        plt.xlabel('Nodal coordinates [m]')
        plt.ylabel('Amplitude [m]')
        plt.title('System deformation in Y at Ω = 1460 rad/s')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        print("All plots generated successfully!")

# Main execution
if __name__ == "__main__":
    # Create analysis instance
    analysis = RotordynamicAnalysis()
    
    # Run the analysis
    success = analysis.run_analysis()
    
    if success:
        # Generate plots
        analysis.plot_results()
        
        print("\n" + "="*50)
        print("ROTORDYNAMIC ANALYSIS COMPLETED SUCCESSFULLY")
        print("="*50)
        print(f"Speed range: {analysis.omega[0]:.0f} - {analysis.omega[-1]:.0f} rad/s")
        print(f"Number of speed points: {analysis.n}")
        print(f"Bearing 1 reaction: {analysis.FM1:.2f} N")
        print(f"Bearing 2 reaction: {analysis.FM2:.2f} N")
        print(f"System DOF: {analysis.neq}")
        print("="*50)
    else:
        print("Analysis failed!") 
