import numpy as np
import random
# from hf_var import stockbarcos



# Las clases que creamos para nuestra versión de Hundir la flota son:

# ----------------------------------
# Clase Barco
# ----------------------------------

class Barco:
    orientaciones = ["N", "S", "E", "O"]

    def __init__(self, nombre, tamaño, cantidad):
        self.nombre = nombre
        self.tamaño = tamaño
        self.cantidad = cantidad
        self.coordenadas = []

    def generar_coordenadas(self, tablero):
        while True:
            fila = random.randint(0, 9)
            columna = random.randint(0, 9)
            orientacion = random.choice(Barco.orientaciones)
            coords_temporales = []

            for i in range(self.tamaño):
                if orientacion == "N":
                    nueva_fila = fila - i
                    nueva_col = columna
                elif orientacion == "S":
                    nueva_fila = fila + i
                    nueva_col = columna
                elif orientacion == "E":
                    nueva_fila = fila
                    nueva_col = columna + i
                else:
                    nueva_fila = fila
                    nueva_col = columna - i

                if nueva_fila < 0 or nueva_fila >= 10 or nueva_col < 0 or nueva_col >= 10:
                    break
                if tablero[nueva_fila, nueva_col] != " ":
                    break

                coords_temporales.append((nueva_fila, nueva_col))

            if len(coords_temporales) == self.tamaño:
                self.coordenadas = coords_temporales
                return coords_temporales

    def colocar_en_tablero(self, tablero):
        for fila, col in self.coordenadas:
            tablero[fila, col] = "O"
        return tablero


# ----------------------------------
# Clase Tablero
# ----------------------------------

class Tablero:
    def __init__(self):
        self.tablero = self.crear_tablero()

    def crear_tablero(self, lado=10):
        return np.full((lado, lado), " ")

    def colocar_barcos(self, stockbarcos):
        for nombre, cantidad in stockbarcos.items():
            tamaño = self.obtener_tamaño_barco(nombre)
            for _ in range(cantidad):
                barco = Barco(nombre, tamaño, cantidad)
                barco.generar_coordenadas(self.tablero)
                barco.colocar_en_tablero(self.tablero)
        return self.tablero

    def obtener_tamaño_barco(self, nombre):
        tamaños = {
            "Destructor": 1,
            "Submarino": 2,
            "Acorazado": 3,
            "Portaaviones": 4
        }
        return tamaños[nombre]

    def mostrar(self):
        print(self.tablero)

    def disparar(self, x, y):
        if self.tablero[x, y] == "O":
            self.tablero[x, y] = "X"
            return "impacto"
        elif self.tablero[x, y] in ["X", "-"]:
            return "repetido"
        else:
            self.tablero[x, y] = "~"
            return "agua"

    def comprobar_victoria(self):
        return not np.any(self.tablero == "O")