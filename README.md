# Simulação LA-VHO vs I-VHO vs D-VHO (VLC/WLAN Hybrid Network)

Reprodução da simulação do artigo:
A. Zeshan e T. Baykas, "Location Aware Vertical Handover in a VLC/WLAN
Hybrid Network," IEEE Access, vol. 9, pp. 129810–129819, 2021.

## Arquivos
- `vho_sim.py` — motor da simulação (movimento do usuário, cobertura VLC,
  bloqueio óptico, e os 3 esquemas de handover: I-VHO, D-VHO, LA-VHO).
- `run_experiments.py` — roda os experimentos (variando Nu e velocidade,
  nas duas salas) e gera as figuras.
- `fig5_nvho.png` — nº médio de handovers vs Nu (equivalente à Fig. 5).
- `fig6_qoe.png` — QoE vs velocidade e vs Nu (equivalente à Fig. 6).
- `fig7_packetloss.png` — perda de pacotes vs velocidade e vs Nu (Fig. 7).

## O que foi implementado fielmente
- Layout: prédio 16m×16m×3m, salas 5.6m×5.6m×3m (4 APs VLC) e
  8m×8m×3m (9 APs VLC), raio de cobertura de cada AP = 1.5m (Tabela 1).
- Movimento do usuário: velocidade 0.2–1 m/s, direção aleatória uniforme
  em [0, 2π], modelo "fluid-flow" (troca de direção periódica).
- Probabilidade de bloqueio óptico crescente e saturante com o número de
  usuários Nu (mesma tendência qualitativa da eq. 15 / Fig. 3 do artigo).
- I-VHO: handover imediato assim que o link óptico cai; volta a VLC assim
  que o link é restabelecido.
- D-VHO: espera um "dwell time" (0.5s ou 1s) antes de comutar para RF.
- LA-VHO: 2 estágios — (1) se Nu ≥ limiar ξ, força WLAN o tempo todo
  (handovers → 0); (2) caso contrário, seleção com pequena histerese que
  representa a minimização do custo de handover via função de custo/
  gradiente descendente do artigo (eq. 17–18, Algoritmo 1).
- Métricas: NVHO (handovers médios/iteração), QoE (com penalidade por
  handover e por "outage" quando tentando VLC bloqueado) e perda de
  pacotes (% do tempo em outage enquanto o modo ativo é VLC).

## Limitações / o que foi simplificado (transparência)
O artigo **não publica** o código-fonte nem todos os parâmetros numéricos
ocultos (pesos exatos w_u/w_s, custo de delay d_c, forma fechada exata de
Pb(Nu) fora da eq. simbólica, etc.). Por isso:
- A relação Pb(Nu) e a função de QoE foram modeladas de forma
  fisicamente coerente, mas calibradas heuristicamente — os **valores
  percentuais exatos do artigo não são reproduzíveis** sem esses
  parâmetros. As **tendências qualitativas**, sim, batem:
  - I-VHO tem sempre o maior número de handovers e a pior QoE (efeito
    ping-pong), mas a menor perda de pacotes.
  - D-VHO com dwell maior (1s) reduz handovers mas aumenta perda de
    pacotes.
  - LA-VHO domina em nº de handovers e QoE; quando Nu ultrapassa o
    limiar ξ, os handovers vão a zero (força WLAN).
- Simulação usa 250 iterações × 400s por ponto (em vez de 1500 × 2100–
  3800s do artigo) para rodar em segundos em vez de horas — os valores
  já convergem estatisticamente para essa escala.

## Como rodar
```bash
python3 run_experiments.py
```
Gera as 3 figuras PNG e imprime uma tabela resumo no terminal.
