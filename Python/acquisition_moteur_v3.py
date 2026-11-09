# ------------------------------------------------------------ #
# Auteur : LOUINEAU Ronan - GE5
# Date : 05/11/26
# Description : acquisition courant / vitesse d'un moteur CC sur un échelon de PWM
#   envoie UNE consigne à l'Arduino (CMD_data.ino v1.4),
#   récupère les mesures brutes, les enregistre dans csv/brut
#   puis les convertit automatiquement dans csv/clean
# ------------------------------------------------------------ #

# -------------------------Librairie---------------------------- #
import argparse
import atexit
import csv
import math
import os
import time
from datetime import datetime

import serial

from convertir_csv_v3 import convertir, DOSSIER_CLEAN

# ---------------------------Paramètres--------------------------- #
PORT = "COM4"           # !!! S'assurer d'être sur le bon port de COM
BAUD = 115200
SEPARATEUR = ","        # mettre ";" pour ouvrir directement le fichier dans Excel (version FR)

F_CLK = 16_000_000      # horloge de l'ATmega2560 (Hz)
TOP = 799               # OCR1A côté Arduino (TN_h)
NB_MAX = 1400           # taille du tampon côté Arduino (NB_MAX)

DOSSIER_BRUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csv", "brut") # dossier des CSV bruts
T_PWM = 2 * TOP / F_CLK # période MLI (mode 9, phase et fréquence correctes) = 99,875 µs

# ------------------------Connexion Arduino----------------------- #
arduino = serial.Serial(PORT, BAUD, timeout=1)
print("Connexion établie avec Arduino")
time.sleep(2)           # la carte redémarre à l'ouverture du port


# ---------------------------Fonction----------------------------- #
def _lire_ligne():
    return arduino.readline().decode(errors="replace").strip()


def _couper_moteur():
    arduino.write(b"0\n")
    _lire_ligne()       # "Nouveau OCR1B = 0"


def modifier_PWM(pourcentage):
    """Consigne PWM directe, sans acquisition : sert à vérifier que le moteur tourne.
    Exemple : modifier_PWM(40) puis modifier_PWM(0)."""
    if not 0 <= pourcentage <= 100:
        print("Erreur : pourcentage hors limites")
        return
    valeur = round(TOP * pourcentage / 100)
    arduino.write(f"{valeur}\n".encode())
    print(_lire_ligne())


def acquisition(pwm_pct, duree_s, repos_s=2.0, dossier_brut=DOSSIER_BRUT, dossier_clean=DOSSIER_CLEAN):
    """Applique un échelon de PWM (en %) à t = 0 et enregistre courant et vitesse
    pendant duree_s secondes. Une seule grandeur est convertie à chaque instant
    d'échantillonnage (courant, vitesse, courant, ...) : chaque grandeur est donc
    échantillonnée toutes les 2 * n_dec périodes MLI. n_dec est le plus petit
    entier qui permet de couvrir la durée demandée.
    repos_s : temps moteur à l'arrêt avant l'échelon.
    Enregistre le CSV brut dans dossier_brut, le convertit dans dossier_clean
    et retourne (chemin du CSV brut, chemin du CSV converti ou None si la conversion échoue)."""
    if not 0 <= pwm_pct <= 100:
        raise ValueError("pwm_pct doit être compris entre 0 et 100")
    if duree_s <= 0:
        raise ValueError("duree_s doit être positif")

    pwm_val = round(TOP * pwm_pct / 100)

    # Plus petite période d'échantillonnage qui tient dans le tampon de l'Arduino
    # (nb mesures de chaque grandeur, une grandeur toutes les 2 * n_dec périodes MLI)
    n_dec = max(1, math.ceil(duree_s / (NB_MAX * 2 * T_PWM)))
    nb = max(1, min(NB_MAX, round(duree_s / (2 * n_dec * T_PWM))))

    # 1) moteur à l'arrêt avant l'échelon
    _couper_moteur()
    time.sleep(repos_s)

    # 2) lancement : l'Arduino acquiert, puis envoie tout après la fin
    arduino.reset_input_buffer()
    arduino.write(f"A,{pwm_val},{n_dec},{nb}\n".encode())
    date = datetime.now()
    print(f"Acquisition en cours : PWM {pwm_pct:g} % (OCR1B = {pwm_val}), "
          f"{nb} points par grandeur, Te = {2 * n_dec * T_PWM * 1e6:.3f} µs")

    # L'entête n'arrive qu'après l'acquisition. On attend avec une échéance, SANS
    # modifier arduino.timeout : reconfigurer le port pendant que l'Arduino envoie
    # les données peut corrompre des octets sous Windows.
    echeance = time.time() + 2 * nb * n_dec * T_PWM + 5
    entete = ""
    while time.time() < echeance and not entete.startswith(("DEBUT", "ERR")):
        entete = _lire_ligne()
    if not entete.startswith("DEBUT"):
        raise RuntimeError(f"Réponse inattendue de l'Arduino : {entete!r}")

    _, pwm_val, n_dec, nb, top = entete.split(",")
    pwm_val, n_dec, nb, top = int(pwm_val), int(n_dec), int(nb), int(top)
    t_conv = n_dec * 2 * top / F_CLK            # temps entre deux conversions (s)
    te = 2 * t_conv                             # période d'échantillonnage de chaque grandeur (s)

    # 3) lecture des mesures brutes
    donnees = []
    for k in range(nb):
        ligne = _lire_ligne()
        champs = ligne.split(",")
        if len(champs) != 2:
            raise RuntimeError(f"Ligne {k + 1}/{nb} invalide : {ligne!r}")
        donnees.append((int(champs[0]), int(champs[1])))

    fin = _lire_ligne().split(",")
    if fin[0] != "FIN" or len(fin) != 4:
        raise RuntimeError(f"Fin de transmission invalide : {fin!r}")
    perdus, duree_us, periodes = int(fin[1]), int(fin[2]), int(fin[3])

    # 4) contrôles
    if perdus:
        print(f"ATTENTION : {perdus} déclenchement(s) arrivé(s) pendant le traitement, "
            f"mesures suspectes. Augmente la durée (période d'échantillonnage plus grande).")
    ecart_us = duree_us - (periodes - 1) * 2 * top / F_CLK * 1e6
    if not 0 <= ecart_us <= 300:
        print(f"ATTENTION : base de temps incohérente (écart de {ecart_us:.0f} µs entre "
            f"micros() et le nombre de périodes MLI).")

    # 5) CSV : temps, mesures brutes, PWM (valeur et %), date de l'essai
    #    le courant est mesuré à k * te, la vitesse t_conv plus tard
    pwm_pct_reel = 100 * pwm_val / top
    nom = f"moteur_PWM{pwm_pct:g}pct_{date:%Y%m%d_%H%M%S}.csv"
    os.makedirs(dossier_brut, exist_ok=True)
    chemin = os.path.join(dossier_brut, nom)
    with open(chemin, "w", newline="", encoding="utf-8") as f:
        ecrivain = csv.writer(f, delimiter=SEPARATEUR)
        ecrivain.writerow(["temps_courant_s", "courant_brut",
                        "temps_vitesse_s", "vitesse_brut",
                        "pwm_valeur", "pwm_pct", "date"])
        for k, (i_brut, w_brut) in enumerate(donnees):
            ecrivain.writerow([f"{k * te:.9f}", i_brut,
                               f"{k * te + t_conv:.9f}", w_brut,
                            pwm_val, f"{pwm_pct_reel:.2f}", f"{date:%Y-%m-%d %H:%M:%S}"])

    print(f"{nb} points enregistrés dans {chemin} "
          f"(Te = {te * 1e6:.3f} µs par grandeur, durée = {(nb - 1) * te + t_conv:.3f} s, "
        f"durée mesurée par l'Arduino = {duree_us / 1e6:.3f} s)")

    # 6) conversion automatique en A et rad/s (le CSV brut est déjà sauvegardé)
    try:
        chemin_clean, _ = convertir(chemin, dossier_clean)
        print(f"CSV converti : {chemin_clean}")
    except Exception as erreur:
        chemin_clean = None
        print(f"ATTENTION : conversion impossible ({erreur}). "
            f"Le CSV brut est conservé : {chemin}")
    return chemin, chemin_clean


def fermer_connexion():
    if arduino.is_open:
        arduino.write(b"0\n")
        arduino.close()
        print("Port série fermé proprement et libéré pour l'IDE.")

# On enregistre cette fonction pour qu'elle s'exécute à la sortie du script
atexit.register(fermer_connexion)


# ----------------------------Lancement--------------------------- #
if __name__ == "__main__":
    parseur = argparse.ArgumentParser(description="Acquisition courant / vitesse sur un échelon de PWM")
    parseur.add_argument("pwm", type=float, nargs="?", help="consigne PWM en %% (0 à 100)")
    parseur.add_argument("duree", type=float, nargs="?", default=1.0,
                        help="durée d'enregistrement en s (défaut : 1.0)")
    parseur.add_argument("--repos", type=float, default=2.0,
                        help="temps moteur à l'arrêt avant l'échelon, en s (défaut : 2.0)")
    args = parseur.parse_args()

    if args.pwm is None:
        print("Mode interactif : acquisition(pwm_pct, duree_s)")
    else:
        acquisition(args.pwm, args.duree, args.repos)