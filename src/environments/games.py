'''
© Edgar Andrade 2018
Email: edgar.andrade@urosario.edu.co

-----------------------------------------------
Games for testing agents.

Includes:
    - Triqui.
-----------------------------------------------
'''
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.offsetbox import AnnotationBbox, OffsetImage, TextArea
import numpy as np
import copy
from time import sleep
from IPython.display import clear_output, SVG, display
from os import path
from pathlib import Path

image_folder = Path.cwd() / Path('environments', 'imagenes')
image_folder.mkdir(parents=True, exist_ok=True)
to_array_folder = Path.cwd() / Path('environments', 'to_array')
to_array_folder.mkdir(parents=True, exist_ok=True)


class Triqui:

    def __init__(self):
        self.estado_inicial = np.matrix([[0]*3]*3)
        self.render_mode = None

    def render(self, estado):
        # Dibuja el tablero correspondiente al estado
        # Input: estado, que es una 3-lista de 3-listas
        fig, axes = plt.subplots(figsize=(5,5))

        # Dibujo el tablero
        step = 1./3
        offset = 0.001
        tangulos = []

        # Borde del tablero
        tangulos.append(patches.Rectangle((0,0),0.998,0.998,\
                                          facecolor='cornsilk',\
                                         edgecolor='black',\
                                         linewidth=2))

        # Creo las líneas del tablero
        for j in range(3):
            locacion = j * step
            # Crea linea horizontal en el rectangulo
            tangulos.append(patches.Rectangle(*[(0, locacion), 1, 0.008],\
                    facecolor='black'))
            # Crea linea vertical en el rectangulo
            tangulos.append(patches.Rectangle(*[(locacion, 0), 0.008, 1],\
                    facecolor='black'))

        for t in tangulos:
            axes.add_patch(t)

        # Cargando imagen de O
        Os = path.join(image_folder, 'Triqui', 'O.png')
        arr_img_O = plt.imread(Os, format='png')
        image_O = OffsetImage(arr_img_O, zoom=0.14)
        image_O.image.axes = axes

        # Cargando imagen de X
        Xs = path.join(image_folder, 'Triqui', 'X.png')
        arr_img_X = plt.imread(Xs, format='png')
        image_X = OffsetImage(arr_img_X, zoom=0.14)
        image_X.image.axes = axes

        offsetX = 0.15
        offsetY = 0.15

        # ASUMO QUE LAS O SE REPRESENTAN CON 1 EN LA MATRIZ
        # Y QUE LAS X SE REPRESENTAN CON 2
        for i in range(3):
            for j in range(3):
                if estado[j, i] == 2:
                    # print("O en (" + str(i) + ", " + str(j) + ")")
                    Y = j
                    X = i
                    # print("(" + str(X) + ", " + str(Y) + ")")
                    ab = AnnotationBbox(
                        image_O,
                        [(X*step) + offsetX, (Y*step) + offsetY],
                        frameon=False)
                    axes.add_artist(ab)
                if estado[j, i] == 1:
                    # print("X en (" + str(i) + ", " + str(j) + ")")
                    Y = j
                    X = i
                    # print("(" + str(X) + ", " + str(Y) + ")")
                    ab = AnnotationBbox(
                        image_X,
                        [(X*step) + offsetX, (Y*step) + offsetY],
                        frameon=False)
                    axes.add_artist(ab)
        axes.axis('off')
        if self.render_mode == 'rgb_array':
            to_array_file = Path(to_array_folder, f'to_array.png')
            plt.savefig(to_array_file)
            return plt.imread(to_array_file)
        else:
            plt.show()

    def player(self, estado):
        # Devuelve el número del jugador a quien corresponde el turno
        # 1 para las X
        # 2 para las O
        num_Xs = np.count_nonzero(estado==1)
        num_Os = np.count_nonzero(estado==2)
        # print("Cantidad O:", num_Os, " Cantidad X:", num_Xs)
        if num_Os < num_Xs:
            return 2
        else:
            return 1

    def acciones(self, estado):
        # Devuelve una lista de parejas que representan las casillas vacías
        # Input: estado, que es una np.matrix(3x3)
        # Output: lista de índices (x,y)
        indices = []
        if np.count_nonzero(estado==0)>0:
            for x in range(3):
                for y in range(3):
                    if estado[y, x] == 0:
                        indices.append((x, y))
        return indices

    def resultado(self, estado, accion):
        # Devuelve el tablero incluyendo una O o X en el accion,
        # dependiendo del jugador que tiene el turno
        # Input: estado, que es una np.matrix(3x3)
        #        accion, de la forma (x,y)
        # Output: estado, que es una np.matrix(3x3)
        assert(accion in self.acciones(estado))
        s = copy.deepcopy(estado)
        x = accion[0]
        y = accion[1]
        s[y, x] = self.player(estado)

        return s

    def es_terminal(self, estado):
        # Devuelve True/False dependiendo si el juego se acabó
        # Input: estado, que es una np.matrix(3x3)
        # Output: objetivo, True/False
        # print("Determinando si no hay casillas vacías...")
        if np.count_nonzero(estado==0)==0:
            return True
        else:
            # print("Buscando triqui horizontal...")
            for y in range(3):
                num_Xs = np.count_nonzero(estado[y,:]==1)
                num_Os = np.count_nonzero(estado[y,:]==2)
                # print("Cantidad O:", num_Os, " Cantidad X:", num_Xs)
                if (num_Os==3) or (num_Xs==3):
                    return True
            # print("Buscando triqui vertical...")
            for x in range(3):
                num_Xs = np.count_nonzero(estado[:,x]==1)
                num_Os = np.count_nonzero(estado[:,x]==2)
                # print("Cantidad O:", num_Os, " Cantidad X:", num_Xs)
                if (num_Os==3) or (num_Xs==3):
                    return True
            # print("Buscando triqui diagonal...")
            if (estado[0,0]==1) and (estado[1,1]==1) and (estado[2,2]==1):
                return True
            elif (estado[0,0]==2) and (estado[1,1]==2) and (estado[2,2]==2):
                return True
            # print("Buscando triqui transversal...")
            if (estado[2,0]==1) and (estado[1,1]==1) and (estado[0,2]==1):
                return True
            elif (estado[2,0]==2) and (estado[1,1]==2) and (estado[0,2]==2):
                return True
        return False

    def utilidad(self, estado, jugador):
		# Devuelve la utilidad del estado donde termina el juego
		# Input: estado, que es una np.matrix(3x3)
		# Output: utilidad, que es un valor -1, 0, 1
		# print("Buscando triqui horizontal
        for y in range(3):
            num_Xs = np.count_nonzero(estado[y,:]==1)
            num_Os = np.count_nonzero(estado[y,:]==2)
            # print("Cantidad O:", num_Os, " Cantidad X:", num_Xs)
            if (num_Os==3):
                return -1
            elif (num_Xs==3):
                return 1
        # print("Buscando triqui vertical...")
        for x in range(3):
            num_Xs = np.count_nonzero(estado[:,x]==1)
            num_Os = np.count_nonzero(estado[:,x]==2)
            # print("Cantidad O:", num_Os, " Cantidad X:", num_Xs)
            if (num_Os==3):
                return -1
            elif (num_Xs==3):
                return 1
        # print("Buscando triqui diagonal...")
        if (estado[0,0]==1) and (estado[1,1]==1) and (estado[2,2]==1):
            return 1
        elif (estado[0,0]==2) and (estado[1,1]==2) and (estado[2,2]==2):
            return -1
        # print("Buscando triqui transversal...")
        if (estado[2,0]==1) and (estado[1,1]==1) and (estado[0,2]==1):
            return 1
        elif (estado[2,0]==2) and (estado[1,1]==2) and (estado[0,2]==2):
            return -1
        # Determina si hay empate
        if np.count_nonzero(estado==0)==0:
            return 0
        return None

    def give_result(self, estado):
        if self.es_terminal(estado):
            playing = self.player(estado)
            utilidad_recibida = self.utilidad(estado, playing)
            if utilidad_recibida == -1:
                print('Juego terminado. ¡Gana O!')
            elif utilidad_recibida == 1:
                    print('Juego terminado. ¡Gana X!')
            else:
                print('Juego terminado. ¡Empate!')
