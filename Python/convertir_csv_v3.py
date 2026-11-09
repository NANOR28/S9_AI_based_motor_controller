# ------------------------------------------------------------ #
# Auteur : LOUINEAU Ronan - GE5
# Date : 05/11/26
# Description : convertit un CSV d'acquisition (valeurs brutes de l'ADC)
#               en CSV propre avec le courant en A et la vitesse en rad/s
#               (courant et vitesse sont mesurés alternativement : chaque
#               grandeur garde sa propre colonne de temps)
# ------------------------------------------------------------ #

# ------------------------------------------------------------ #
# Utilisation :
#   Automatique : acquisition_moteur_v2.py appelle convertir() à la fin de chaque
#   acquisition (rien à faire).
#
#   Manuel (pour reconvertir un ancien fichier brut) :
#   1) python convertir_csv_v2.py
#   2) coller le nom du fichier à convertir quand le programme le demande
#      (ou le glisser-déposer dans la fenêtre), puis Entrée
#   3) le fichier converti est créé dans csv/clean, avec "clean_csv_"
#      devant le nom d'origine :
#        csv/brut/moteur_PWM50pct_20261105_120152.csv
#        -> csv/clean/clean_csv_moteur_PWM50pct_20261105_120152.csv
#   Entrée sans rien écrire : quitter.
# ------------------------------------------------------------ #

# -------------------------Librairie---------------------------- #
import csv
import os

# ---------------------------Paramètres--------------------------- #
SEPARATEUR = ","        # le même que dans acquisition_moteur_v2.py
# dossier des CSV convertis : csv/clean à côté de ce script (créé si absent)
DOSSIER_CLEAN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csv", "clean")
VREF = 5.0              # tension de référence de l'ADC (AVcc) en V
RESOLUTION = 1023       # ADC 10 bits : 0..1023
GAIN_COURANT = 10.0     # capteur de courant : 10 V/A
GAIN_VITESSE = 0.5      # capteur de vitesse : 1 V/rad/s --> changé pour eviter la saturation


# ---------------------------Fonction----------------------------- #
def convertir(chemin_entree, dossier_sortie=DOSSIER_CLEAN):
    """Lit un CSV brut et crée clean_csv_<nom d'origine> dans dossier_sortie
    (csv/clean par défaut), avec courant_A et vitesse_rad_s à la place des valeurs brutes.
    Les colonnes temps_courant_s, temps_vitesse_s, pwm_valeur, pwm_pct et date
    sont conservées."""
    nom = os.path.basename(chemin_entree)
    os.makedirs(dossier_sortie, exist_ok=True)
    chemin_sortie = os.path.join(dossier_sortie, "clean_csv_" + nom)

    with open(chemin_entree, newline="", encoding="utf-8") as f:
        lecteur = csv.DictReader(f, delimiter=SEPARATEUR)
        lignes = list(lecteur)
        colonnes = lecteur.fieldnames or []

    colonnes_brutes = ("temps_courant_s", "courant_brut", "temps_vitesse_s", "vitesse_brut")
    for colonne in colonnes_brutes:
        if colonne not in colonnes:
            raise ValueError(f"colonne '{colonne}' absente : ce n'est pas un fichier brut "
                            f"d'acquisition (colonnes trouvées : {colonnes})")

    autres = [c for c in colonnes if c not in colonnes_brutes]
    volt_par_pas = VREF / RESOLUTION

    with open(chemin_sortie, "w", newline="", encoding="utf-8") as f:
        ecrivain = csv.writer(f, delimiter=SEPARATEUR)
        ecrivain.writerow(["temps_courant_s", "courant_A",
                        "temps_vitesse_s", "vitesse_rad_s"] + autres)
        for ligne in lignes:
            courant = int(ligne["courant_brut"]) * volt_par_pas / GAIN_COURANT
            vitesse = int(ligne["vitesse_brut"]) * volt_par_pas / GAIN_VITESSE
            ecrivain.writerow([ligne["temps_courant_s"], f"{courant:.6f}",
                            ligne["temps_vitesse_s"], f"{vitesse:.4f}"]
                            + [ligne[c] for c in autres])

    return chemin_sortie, len(lignes)


# ----------------------------Lancement--------------------------- #
if __name__ == "__main__":
    print("Conversion des CSV bruts en A et rad/s")
    while True:
        # Les guillemets sont retirés : "Copier en tant que chemin" de Windows en ajoute
        saisie = input("\nNom du fichier à convertir (Entrée pour quitter) : ").strip().strip('"').strip("'")
        if not saisie:
            break
        if not os.path.isfile(saisie):
            print(f"Fichier introuvable : {saisie}\n"
                f"(dossier actuel : {os.getcwd()} - lance le script depuis le dossier du fichier "
                f"ou colle son chemin complet)")
            continue
        if os.path.basename(saisie).startswith("clean_csv_"):
            print("Ce fichier est déjà converti.")
            continue
        try:
            sortie, nb = convertir(saisie)
            print(f"{nb} points convertis -> {sortie}")
        except Exception as erreur:
            print(f"Erreur : {erreur}")