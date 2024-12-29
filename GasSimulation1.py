# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 22:34:08 2024

@author: 90545
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation # animasyon için
from scipy.spatial.distance import pdist, squareform 
import matplotlib.animation as animation

print(animation.writers)

"""
Küçük ölçeklerle çalışmak, hesaplama zorluğu yaratıyor ve sayısal stabiliteyi etkileyebiliyor. 
Gerçek birimlerden bağımsız çalıştırılmalı.
Birim ölçeğini 1e6 olarak seçiyorum.
Uzaklığı ölçeklendiriyorsak zamanı de ölçeklendirmemiz lazım.
"""

distScale = 1e6
timeScale = 1e9
velScale = distScale/timeScale

r = 2.02e-9
half_slit = 0.08
m = 1
number_of_particles = 600
positions = np.random.random((number_of_particles, 2)) * np.array([0.4, 0.9]) # x max *.4, y max 1 olacak şekilde
positions = np.abs(positions)

mean_velocity = 150 * velScale
random_angle = np.random.random(number_of_particles) * 2 * np.pi
velocity = np.array([np.cos(random_angle), np.sin(random_angle)]).T * mean_velocity

# mean_velocity = 150 * velScale
# random_angle = (np.random.random(number_of_particles)*0.8+0.1) * 2 * np.pi
# velocity = np.array([np.cos(random_angle), np.sin(random_angle)]).T * mean_velocity

# # Hızların 0'dan büyük olmasını sağlamak için
velocity[velocity == 0] += np.finfo(float).eps  # Eğer hız 0 ise, çok küçük bir değeri ekle

# # velocity = np.abs(velocity)

"""
pdist, noktalar arası uzaklıkları hesaplayıp vektör döndürür.
squareform, pdist tarafından üretiler vektörü matrise döndürüyor
"""

import matplotlib.patches as patches
import matplotlib.path as path

x_coordinate, y_coordinate = 0, 1
class mySimulation:
    def __init__(self, r, m, position, velocity, half_slit):
        self.r = r
        self.m = m
        
        self.half_slit = half_slit
        
        self.position = np.asarray(position, dtype=float) # arraysa yeni kopya oluşturmuyor!!!
        self.velocity = np.asarray(velocity, dtype=float)
        
        # Parçacık sayısı, hız ve pozisyon arrayinin uzunluğunda
        self.num = self.position.shape[0]
        
        self.n = 0
    
    def forward(self,dt):
        # Adım sayısını artır
        self.n += 1
         
        # Hızlar aynı kalırsa, position = velocity * dt olur.
        self.position += self.velocity * dt
        
        # Çarpışmanın gerçekleşmesi için dist < 2r olmalı.
        #1-2, 1-3, 2-3... tüm uzunlukları bulur.
        dist_array = pdist(self.position)
        dist_matrix = squareform(dist_array)
        
        # # dist < 2*self.r sağlayanları (çarpışanları) bul.
        # i_arr, j_arr = np.where(dist_matrix < 2*self.r)
        
        # # ij parçarsa ji de çarpar aslında. Yani bu durumu bir kere
        # # almak için i>j veyahut j>i koşulunu koymam gerekir.
        # k = i_arr < j_arr # True, False arrayi oluşturdu.
        # i_arr, j_arr = i_arr[k], j_arr[k]
        
        colliding_indices = np.where(dist_matrix < 2 * self.r)
        particle_pairs = [(i, j) for i, j in zip(*colliding_indices) if i < j]
        # for i,j in zip(i_arr, j_arr):
        for i,j in particle_pairs:
      
            i_position = self.position[i]
            j_position = self.position[j]
            i_velocity = self.velocity[i]
            j_velocity = self.velocity[j]
            
            
            relative_position = i_position - j_position
            relative_velocity = i_velocity - j_velocity
            
            relative_position_magnitude = np.dot(relative_position, relative_position)
            relative_velocity_proj = np.dot(relative_velocity, relative_position)
            relative_velocity_proj = 2*relative_position*relative_velocity_proj / relative_position_magnitude - relative_velocity
                        
            # https://en.wikipedia.org/wiki/Elastic_collision
            v_cm = (i_velocity+j_velocity) / 2
            self.velocity[i] = v_cm - relative_velocity_proj/2
            self.velocity[j] = v_cm + relative_velocity_proj/2
        
        wall_l = self.position[:, x_coordinate] < self.r
        wall_r = self.position[:, x_coordinate] > 1-self.r
        wall_u = self.position[:, y_coordinate] > 1-self.r
        wall_d = self.position[:, y_coordinate] < self.r 
        
        # Çarpışma gerçekleştiyse -1 ile çarpıyorum hızımı, çünkü yön değiştirecek.
        self.velocity[wall_l | wall_r, x_coordinate] *= -1
        self.velocity[wall_u | wall_d, y_coordinate] *= -1
        
        slit_wall = ( ((self.position[:, y_coordinate]<0.5-self.half_slit) | (self.position[:, y_coordinate]>0.5+self.half_slit)) 
                     & ((self.position[:, x_coordinate]<0.5+self.half_slit/2) & (self.position[:, x_coordinate]>0.5-self.half_slit/2)) )
        
        self.velocity[slit_wall] *= -1



mysim = mySimulation(r*distScale, m, positions, velocity, half_slit)

# Create the figure and axis for plotting
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# Duvar ve slit
wall_top = patches.Rectangle((0.475, 0.5 + half_slit), 0.05, 0.5 - half_slit, linewidth=1, edgecolor='r', facecolor='r')
wall_bottom = patches.Rectangle((0.475, 0), 0.05, 0.5 - half_slit, linewidth=1, edgecolor='r', facecolor='r')

ax.add_patch(wall_top)
ax.add_patch(wall_bottom)

# Scatter plot for particles
scatter = ax.scatter(mysim.position[:, 0], mysim.position[:, 1])


room_count_left = np.array([])
room_count_right = np.array([])
time_series = np.array([])

def update(frame):
    # Değişkenleri global olarak tanımladım
    global room_count_left, room_count_right, time_series

    left_room_counter = np.sum(mysim.position[:, 0] < 0.5)
    right_room_counter = number_of_particles - left_room_counter

    time_series = np.append(time_series, 1/30 * frame)    
    room_count_left = np.append(room_count_left, left_room_counter)
    room_count_right = np.append(room_count_right, right_room_counter)
        
    mysim.forward(1/30)  # Move the simulation forward by a small timestep
    scatter.set_offsets(mysim.position)  # Update the positions of the particles
    return scatter,


# fig oluşturuldu, update her framede çağırılan fonksiyon, her kare hızı
ani = FuncAnimation(fig, update, frames=4000, interval=10, blit=True)
ani.save('simulation.gif', writer='ffmpeg', fps=30)

plt.figure(figsize=(10, 6))
plt.plot(time_series, room_count_left, label='Left Room', color='blue')
plt.plot(time_series, room_count_right, label='Right Room', color='green')
plt.xlabel('Zaman')
plt.ylabel('Num of Particles')
plt.title('Particle Distribution.')
plt.legend()
plt.grid()

plt.show()