# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import random
from matplotlib import pyplot as plt
import math
import sys


filename = sys.argv[1]
#filename = 'data.txt'

with open(file=filename, mode="r") as file:
    data = list(map(int, file.read().split(" ")))

    a=data[0]
    b=data[1]
    step = data[2]
    energylist = data[4:]



    for energy in energylist:
        demon_energy = energy

        df = pd.DataFrame(index=np.arange(a), columns=np.arange(b))
            
        #buduje macierz wypełniam jedynkami
        for i in range(len(df)):
           for j in range(len(df.columns)):
               #df.at[i, j] = random.randint(0, 1)
               df.at[i, j] = 1
               if df.at[i, j] == 0:
                   df.at[i, j] = -1
        

        demon_energy_tab = []
        board_energy_tab = []
        board_energy = -(a * b) * 2
        magnetization_in_step = []
        
        #losuje index jesli indeks value jest jedynka to zmieniam na minus jeden a jak minus jeden to jeden
        for i in range(step):
                random1 = random.randint(0, a-1)
                random2 = random.randint(0, b-1)

                

                demon_energy_tab.append(demon_energy)
                board_energy_tab.append(board_energy)
                
                energy1 = 0
                energy2 = 0
        
                if random1 - 1 < 0: 
                    left = (a - 1) 
                else:
                    left = random1 - 1
                if random1 + 1 >= a:
                    right = 0 
                else: 
                    right = random1 + 1
                if random2 - 1 < 0:
                    top = (b - 1) 
                else: 
                    top = random2 - 1
                if random2 + 1 >= b:
                    bottom = 0
                else: 
                    bottom = random2 + 1
                
                
                energy1 = -(df[random1][random2]) * (df[left][random2] + df[right][random2] + df[random1][top] + df[random1][bottom])
                
                
                if random1 - 1 < 0: 
                    left = (a - 1) 
                else:
                    left = random1 - 1
                if random1 + 1 >= a:
                    right = 0 
                else: 
                    right = random1 + 1
                if random2 - 1 < 0:
                    top = (b - 1) 
                else: 
                    top = random2 - 1
                if random2 + 1 >= b:
                    bottom = 0
                else: 
                    bottom = random2 + 1
                
                
                energy2 = df[random1][random2] * (df[left][random2] + df[right][random2] + df[random1][top] + df[random1][bottom])
                
                
                diff = energy2 - energy1
                
                
        
                if diff <= 0 or (0 < diff <= demon_energy):
                    df[random1][random2] = -df[random1][random2]
                    demon_energy -= diff
                    board_energy += diff
                     
                mag = 0
        
                for w in range(len(df)):
                  for j in range(len(df.columns)):
                        mag += df.at[w, j]
                
 
                mag = mag/(a*b)
                magnetization_in_step.append(mag)


        fig, axs = plt.subplots(2, 2)
        axs[0, 0].plot(board_energy_tab)
        axs[0, 0].set_title("Board energy")
        axs[1, 0].plot(board_energy_tab)
        axs[1, 0].set_title("Board energy")
        axs[1, 0].sharex(axs[0, 0])
        axs[0, 1].plot(demon_energy_tab)
        axs[0, 1].set_title("Demon energy")
        axs[1, 1].plot(magnetization_in_step)
        axs[1, 1].set_title("Magnetization")
        fig.tight_layout()
                
        plt.show()










     
