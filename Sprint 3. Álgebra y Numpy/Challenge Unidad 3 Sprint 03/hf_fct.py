# Las funciones del juego
import numpy as np
import random

def pedir_coordenadas():
    """
    Pide al usuario que introduzca coordenadas válidas.
    """
    while True:
        try:
            x = int(input("Introduce la fila (0-9): "))
            y = int(input("Introduce la columna (0-9): "))
            if 0 <= x <= 9 and 0 <= y <= 9:
                return x, y
            else:
                print("Coordenadas fuera de rango. Intenta de nuevo.")
        except ValueError:
            print("Entrada no válida. Debes introducir números del 0 al 9.")



def mostrar_tablero_cpu_visible(tablero_cpu, tablero_visible):
    """
    Muestra el tablero de la CPU solo con los disparos recibidos (X y -).
    """
    for i in range(10):
        for j in range(10):
            if tablero_cpu.tablero[i, j] == "X" or tablero_cpu.tablero[i, j] == "~":
                tablero_visible[i, j] = tablero_cpu.tablero[i, j]
    print("Tablero de la CPU (solo impactos visibles):")
    print(tablero_visible)