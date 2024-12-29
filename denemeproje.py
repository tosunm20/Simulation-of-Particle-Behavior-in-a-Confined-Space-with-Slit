# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 13:40:14 2024

@author: 90545
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial.distance import pdist, squareform
import matplotlib.patches as patches
import matplotlib.animation as animation

# Ölçekler
distScale = 1e6
timeScale = 1e9
velScale = distScale / timeScale

# Sabitler
r = 2.02e-9
half_slit = 0.2
m = 1
number_of_particles = 400

# Başlangıç pozisyonları
positions = np.random.random((number_of_particles, 2)) * np.array([0.4, 0.9])
positions = np.abs(positions)

#Başlangıç hızları
mean_velocity = 200 * velScale
random_angle = np.random.random(number_of_particles) * 2 * np.pi
velocity = np.array([np.cos(random_angle), np.sin(random_angle)]).T * mean_velocity

x_coordinate, y_coordinate = 0, 1
class mySimulation:
    def __init__(self, radius, mass, initial_positions, initial_velocities, slit_half_width):
        self.radius = radius
        self.mass = mass
        self.slit_half_width = slit_half_width

        self.positions = np.array(initial_positions, dtype=float)
        self.velocities = np.array(initial_velocities, dtype=float)

        self.num_particles = self.positions.shape[0]
        self.time_step_counter = 0

    def forward(self, dt):
        self.time_step_counter += 1
        self.positions += self.velocities * dt

        # Çarpışma kontrolü
        pairwise_distances = pdist(self.positions)
        distance_matrix = squareform(pairwise_distances)

        colliding_indices = np.where(distance_matrix < 2 * self.radius)
        particle_pairs = [(i, j) for i, j in zip(*colliding_indices) if i < j]

        for i, j in particle_pairs:
            pos_i = self.positions[i]
            vel_i = self.velocities[i]
            pos_j = self.positions[j]
            vel_j = self.velocities[j]

            relative_pos = pos_i - pos_j
            relative_vel = vel_i - vel_j

            pos_magnitude_squared = np.dot(relative_pos, relative_pos)
            velocity_projection = np.dot(relative_vel, relative_pos)
            velocity_projection = 2 * relative_pos * velocity_projection / pos_magnitude_squared - relative_vel


            center_of_mass_velocity = (vel_i + vel_j) / 2
            self.velocities[i] = center_of_mass_velocity - velocity_projection / 2
            self.velocities[j] = center_of_mass_velocity + velocity_projection / 2

        # Duvarlarla çarpışma
        for dim, (lower_bound, upper_bound) in enumerate([(self.radius, 1 - self.radius)]):
            outside_lower = self.positions[:, dim] < lower_bound
            outside_upper = self.positions[:, dim] > upper_bound
            self.velocities[outside_lower | outside_upper, dim] *= -1

        # Slit çarpışması
        slit_condition = (
            ((self.positions[:, y_coordinate] < 0.5 - self.slit_half_width) |
             (self.positions[:, y_coordinate] > 0.5 + self.slit_half_width)) &
            ((0.5 - self.slit_half_width / 2 < self.positions[:, x_coordinate]) &
             (self.positions[:, x_coordinate] < 0.5 + self.slit_half_width / 2))
        )
        self.velocities[slit_condition] *= -1


mysim = mySimulation(r * distScale, m, positions, velocity, half_slit)

# Plot oluşturma
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# Duvar ve slit
wall_top = patches.Rectangle((0.475, 0.5 + half_slit), 0.05, 0.5 - half_slit, linewidth=1, edgecolor='r', facecolor='r')
wall_bottom = patches.Rectangle((0.475, 0), 0.05, 0.5 - half_slit, linewidth=1, edgecolor='r', facecolor='r')

ax.add_patch(wall_top)
ax.add_patch(wall_bottom)

# Parçacıkların scatter grafiği
scatter = ax.scatter(mysim.positions[:, 0], mysim.positions[:, 1])

room_count_left = np.array([])
room_count_right = np.array([])
time_series = np.array([])

def update(frame):
    global room_count_left, room_count_right, time_series

    left_room_counter = np.sum(mysim.positions[:, 0] < 0.5)
    right_room_counter = number_of_particles - left_room_counter

    time_series = np.append(time_series, 1 / 30 * frame)    
    room_count_left = np.append(room_count_left, left_room_counter)
    room_count_right = np.append(room_count_right, right_room_counter)
        
    mysim.forward(1 / 30)  # Simülasyonu bir adım ileri taşı
    scatter.set_offsets(mysim.positions)  # Parçacık pozisyonlarını güncelle
    return scatter,


ani = FuncAnimation(fig, update, frames=4000, interval=10, blit=True)
ani.save('simulation.gif', writer='ffmpeg', fps=30)

# Zaman serisi grafiği
plt.figure(figsize=(10, 6))
plt.plot(time_series, room_count_left, label='Left Room', color='blue')
plt.plot(time_series, room_count_right, label='Right Room', color='green')
plt.xlabel('Time')
plt.ylabel('Number of Particles')
plt.title('Particle Distribution')
plt.legend()
plt.grid()

plt.show()
