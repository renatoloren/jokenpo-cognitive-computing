import numpy as np
import cv2
from matplotlib import pyplot as plt

# Define as imagens a serem detectadas
img_pedra = cv2.imread("pedra.png", cv2.IMREAD_GRAYSCALE)
img_papel = cv2.imread("papel.png", cv2.IMREAD_GRAYSCALE)
img_tesoura = cv2.imread("tesoura.png", cv2.IMREAD_GRAYSCALE)

# Define o SIFT
sift = cv2.SIFT_create()

# Detecta os keypoints das imagens das maos
kp1, des_pedra = sift.detectAndCompute(img_pedra, None)
kp2, des_papel = sift.detectAndCompute(img_papel, None)
kp3, des_tesoura = sift.detectAndCompute(img_tesoura, None)

# Inicializa o vídeo
cap = cv2.VideoCapture("pedra-papel-tesoura.mp4")

class Jogador:
    def __init__(self, nome):
        self.nome = nome
        self.mao = 'o'
        self.ultimaMao = 'x'
        self.pontos = 0
    
j1 = Jogador('Renato')
j2 = Jogador('Leo')

#variaveis de controle
metadeFrame = 0
MIN_MATCHES = 10

#exibe os textos de placar
def placar():
    cv2.putText(frame, f'{j1.pontos} X {j2.pontos}', (metadeFrame - 50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)
    cv2.putText(frame, f'{j1.nome}: {j1.mao}', (metadeFrame - 610, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)
    cv2.putText(frame, f'{j2.nome}: {j2.mao}', (metadeFrame + 280, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)
    cv2.putText(frame, f'{winner}', (metadeFrame - 50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)

def identificar_jogadores(matchs_pedra, matchs_tesoura, matchs_papel):
        if len(matchs_pedra) > MIN_MATCHES: 
            if len(matchs_tesoura) > MIN_MATCHES or len(matchs_papel) > MIN_MATCHES:
                #Define se é o jogador 1 ou 2 pelo lado do jogo
                if kp[matchs_pedra[0].trainIdx].pt[0] < metadeFrame:
                        j1.mao = 'pedra'
                else:
                        j2.mao = 'pedra' 
            else:
                j1.mao = 'pedra' 
                j2.mao = 'pedra'

        if len(matchs_papel) > MIN_MATCHES: 
            if len(matchs_tesoura) > MIN_MATCHES or len(matchs_pedra) > MIN_MATCHES:
                if kp[matchs_papel[0].trainIdx].pt[0] < metadeFrame:
                        j1.mao = 'papel'
                else:
                        j2.mao = 'papel' 
            else:
                j1.mao = 'papel'
                j2.mao = 'papel'

        if len(matchs_tesoura) > MIN_MATCHES: 
            if len(matchs_papel) > MIN_MATCHES or len(matchs_pedra) > MIN_MATCHES:
                if kp[matchs_tesoura[0].trainIdx].pt[0] < metadeFrame:
                        j1.mao = 'tesoura'
                else:
                        j2.mao = 'tesoura'
            else:
                j1.mao = 'tesoura'
                j2.mao = 'tesoura'

#Função que define quem é o ganhador
def jogo(j1, j2):
    if j1.mao == j2.mao:
        return 'empate'

    if j1.mao == 'pedra' and j2.mao == 'tesoura' or \
        j1.mao == 'tesoura' and j2.mao == 'papel' or \
        j1.mao == 'papel' and j2.mao == 'pedra':
            j1.pontos += 1
            return f'{j1.nome} venceu'
    
    if j2.mao == 'pedra' and j1.mao == 'tesoura' or \
        j2.mao == 'tesoura' and j1.mao == 'papel' or \
        j2.mao == 'papel' and j1.mao == 'pedra':
            j2.pontos += 1
            return f'{j2.nome} venceu'
     

count = 0
while(cap.isOpened()):

    ret, frame = cap.read()
    count += 1

    if not ret:
        break

    # Controlador para ler 10 em 10 frames para melhorar desempenho
    if count % 10 != 0:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    metadeFrame = int(gray.shape[1]/2)
    kp, des = sift.detectAndCompute(gray, None)
    bf = cv2.BFMatcher()

    matches1 = bf.knnMatch(des_pedra, des, k=2)
    matches2 = bf.knnMatch(des_papel, des, k=2)
    matches3 = bf.knnMatch(des_tesoura, des, k=2)

    matchs_pedra = []
    for m,n in matches1:
        if m.distance < 0.75 * n.distance:
            matchs_pedra.append(m)

    matchs_papel = []        
    for m,n in matches2:
        if m.distance < 0.75 * n.distance:
            matchs_papel.append(m)

    matchs_tesoura = []        
    for m,n in matches3:
        if m.distance < 0.75 * n.distance:
            matchs_tesoura.append(m)

    j1.ultimaMao = j1.mao
    j2.ultimaMao = j2.mao

    identificar_jogadores(matchs_pedra, matchs_tesoura, matchs_papel)

    if (j1.ultimaMao != j1.mao or j2.ultimaMao != j2.mao):
        winner = jogo(j1, j2)
    
    placar()
    
    cv2.imshow("Frame", frame)
    # Se a tecla 'q' for pressionada, encerra o loop
    if cv2.waitKey(20) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()