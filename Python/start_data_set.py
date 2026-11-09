# ------------------------------------------------------------ #
# Auteur : LOUINEAU Ronan - GE5
# Date : 09/11/26
# Fichier : campagne_essais_moteur.py
# Description : Automatisation du balayage de la PWM de 10% à 100%
# ------------------------------------------------------------ #

import time

# Import de la fonction principale depuis votre script
from acquisition_moteur_v3 import acquisition 

# ------------------------------------------------------------ #
# --- Paramètres de la campagne ---
PWM_DEBUT = 20
PWM_FIN = 100
PAS_PWM = 2

DUREE_MESURE_S = 1.5  # Durée d'acquisition pour chaque palier
REPOS_ENTRE_ESSAIS_S = 0  # Temps pour laisser le moteur s'arrêter complètement
# ------------------------------------------------------------ #

def lancer_campagne():
    print(f"=== Début de la campagne d'essais : PWM de {PWM_DEBUT}% à {PWM_FIN}% ===")
    
    for pwm in range(PWM_DEBUT, PWM_FIN + 1, PAS_PWM):
        print(f"\n[{time.strftime('%H:%M:%S')}] --- Lancement de l'essai pour PWM = {pwm}% ---")
        
        try:
            # Appel direct de votre fonction existante
            # Le port série reste ouvert et réutilisé à chaque itération
            acquisition(
                pwm_pct=pwm, 
                duree_s=DUREE_MESURE_S, 
                repos_s=REPOS_ENTRE_ESSAIS_S
            )
            
        except Exception as e:
            print(f"Erreur inattendue lors de l'essai à {pwm}% : {e}")
            print("Poursuite avec la valeur suivante...")
            
    print("\n=== Campagne d'essais terminée avec succès ===")
    print("Tous les fichiers CSV ont été générés dans les dossiers correspondants.")

if __name__ == "__main__":
    lancer_campagne()