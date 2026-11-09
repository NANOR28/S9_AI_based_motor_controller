# ------------------------------------------------------------ #
# Auteur : LOUINEAU Ronan - GE5
# Date : 07/11/26
# Description : prétraitement de la campagne d'essais moteur CC (échelons de PWM)
#               PWM, remet courant et vitesse sur la même grille de temps, ajuste un
#               1er ordre (tau, valeur en régime établi) et exporte :
#                 csv/filtre/filtre_<essai>.csv      une série filtrée par essai
#                 dataset/resume_campagne.csv        une ligne par essai (PWM, I_ss, w_ss, tau...)
#                 dataset/dataset_campagne.npz       toutes les séries (tableaux NumPy)
#                 figures/*.png                      avant/après par essai + synthèse
# ------------------------------------------------------------ #

# -------------------------Librairie---------------------------- #
import argparse
import glob
import os
import re

import matplotlib

matplotlib.use("Agg")  # figures enregistrées en PNG, pas de fenêtre
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import butter, savgol_filter, sosfiltfilt

# ---------------------------Paramètres--------------------------- #
RACINE = os.path.dirname(os.path.abspath(__file__))
DOSSIER_ENTREE = os.path.join(RACINE, "csv", "clean")

T_PRE_S = 0  # temps moteur à l'arrêt enregistré avant l'échelon (à supprimer)
METHODE = "butter"  # "butter" (Butterworth zéro-phase) ou "savgol" (Savitzky-Golay)
FC_HZ = 15.0  # coupure du passe-bas (Butterworth). Le courant a une oscillation vers 40 Hz
#               (mesurée sur l'essai à 95 %) : fc = 40 Hz la laisse passer. Plus fc est basse,
#               plus le signal est lisse, mais plus le pic de courant du démarrage est atténué.
#               Le moteur (tau ~ 60 ms, soit ~3 Hz) reste très en dessous de 15 Hz.
ORDRE = 2  # ordre du Butterworth (appliqué aller-retour : pente effective doublée)
FENETRE = 41  # fenêtre en points (Savitzky-Golay, polynôme d'ordre 2)
FRAC_REGIME = 0.30  # fraction finale de l'enregistrement utilisée comme régime établi

LSB_I = 5 / 1023 / 10  # pas de quantification du courant (A)
LSB_W = 5 / 1023  # pas de quantification de la vitesse (rad/s)
W_PLEIN_ECHELLE = 5.0  # vitesse correspondant à 5 V sur l'ADC (rad/s)
SEUIL_DEMARRAGE_I = 0.01  # A    : détection du début de l'échelon (contrôle)
SEUIL_DEMARRAGE_W = 0.01  # rad/s

# Couleurs des figures (palette de référence, fond clair)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BRUT, FILTRE, AJUST = "#a8a7a2", "#2a78d6", "#eb6834"


# ---------------------------Fonctions---------------------------- #
def charger(chemin):
    df = pd.read_csv(chemin)
    if df.shape[1] == 1:  # fichier avec séparateur ";"
        df = pd.read_csv(chemin, sep=";")
    colonnes = {"temps_courant_s", "courant_A", "temps_vitesse_s", "vitesse_rad_s", "pwm_pct"}
    manquantes = colonnes - set(df.columns)
    if manquantes:
        raise ValueError(f"colonnes manquantes : {sorted(manquantes)} (CSV de csv/clean attendu)")
    return df


def pwm_nominal(nom_fichier, pwm_reel):
    m = re.search(r"PWM(\d+(?:\.\d+)?)pct", nom_fichier)
    return float(m.group(1)) if m else float(round(pwm_reel))


def filtrer(x, fs, methode=METHODE, fc=FC_HZ, ordre=ORDRE, fenetre=FENETRE):
    """Filtre sans déphasage (aller-retour), donc sans retard : adapté à la préparation
    du dataset, pas à un filtre temps réel."""
    if methode == "savgol":
        fen = fenetre if fenetre % 2 else fenetre + 1
        fen = min(fen, len(x) if len(x) % 2 else len(x) - 1)
        return savgol_filter(x, fen, 2, mode="interp")
    fc = min(fc, 0.45 * fs)
    sos = butter(ordre, fc, btype="low", fs=fs, output="sos")
    return sosfiltfilt(sos, x)


def modele(t, y_inf, y0, tau):
    """1er ordre : croissant (y0 < y_inf) ou décroissant (y0 > y_inf)."""
    return y_inf + (y0 - y_inf) * np.exp(-t / tau)


def ajuster(t, y, y_inf0, y00, tau0=0.05):
    """Ajustement moindres carrés d'un 1er ordre. Retourne (y_inf, y0, tau, r2) ou None."""
    try:
        popt, _ = curve_fit(
            modele,
            t,
            y,
            p0=[y_inf0, y00, tau0],
            bounds=([-np.inf, -np.inf, 1e-4], [np.inf, np.inf, 10.0]),
            maxfev=20000,
        )
    except Exception:
        return None
    ss_res = np.sum((y - modele(t, *popt)) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return (*popt, r2)


def stable(x, lsb):
    """Régime établi : les deux moitiés de la fenêtre ont la même moyenne (à 3 % ou 3 LSB près)."""
    moitie = len(x) // 2
    ecart = abs(x[moitie:].mean() - x[:moitie].mean())
    return bool(ecart <= max(0.03 * abs(x.mean()), 3 * lsb))


def traiter_fichier(chemin, methode, fc, ordre, fenetre, t_pre, frac):
    nom = os.path.basename(chemin)
    df = charger(chemin)
    tc = df["temps_courant_s"].to_numpy(float)
    i_brut = df["courant_A"].to_numpy(float)
    tv = df["temps_vitesse_s"].to_numpy(float)
    w_brut = df["vitesse_rad_s"].to_numpy(float)

    te = float(np.median(np.diff(tc)))
    fs = 1.0 / te

    # 1) suppression des 0,5 s à l'arrêt (nb_retard côté Arduino = round(0,5 / Te)).
    #    On coupe AVANT de filtrer : un filtre aller-retour étalerait la marche de
    #    l'échelon de part et d'autre de t = 0 (courant et vitesse faux au démarrage).
    n_drop = int(round(t_pre / te))
    if n_drop >= len(tc) - 10:
        raise ValueError("enregistrement trop court par rapport au temps de repos à supprimer")
    actifs = np.flatnonzero((i_brut > SEUIL_DEMARRAGE_I) | (w_brut > SEUIL_DEMARRAGE_W))
    if t_pre > 0 and actifs.size and abs(int(actifs[0]) - n_drop) > 3:
        print(
            f"  ATTENTION : début d'échelon détecté à l'échantillon {actifs[0]} "
            f"alors que {n_drop} sont retirés ({nom})."
        )
    t0 = tc[n_drop]
    t = tc[n_drop:] - t0  # grille du courant, t = 0 au début de l'échelon
    t_w = tv[n_drop:] - t0  # grille de la vitesse (décalée d'une demi-période)
    i_b, w_b_nat = i_brut[n_drop:], w_brut[n_drop:]

    # 2) filtrage de chaque grandeur sur sa propre grille (le bord de départ est
    #    prolongé par symétrie impaire, ce qui suit la tendance au lieu de la lisser)
    i_f = filtrer(i_b, fs, methode, fc, ordre, fenetre)
    w_f_nat = filtrer(w_b_nat, fs, methode, fc, ordre, fenetre)

    # 3) vitesse ramenée sur la grille de temps du courant (interpolation linéaire)
    w_b = np.interp(t, t_w, w_b_nat)
    w_f = np.interp(t, t_w, w_f_nat)

    # 4) régime établi : moyenne de la fin de l'enregistrement
    t_regime = (1 - frac) * t[-1]
    m = t >= t_regime
    i_ss, w_ss = float(i_b[m].mean()), float(w_b[m].mean())
    ond_eff = float(np.std(i_b[m] - i_f[m]))
    tourne = w_ss > 0.05
    sature = bool(tourne and np.mean(w_b[m] >= W_PLEIN_ECHELLE - 2 * LSB_W) > 0.5)
    etabli = bool(stable(i_b[m], LSB_I) and stable(w_b[m], LSB_W))
    utilisable = bool(tourne and not sature and etabli)

    # 5) ajustement 1er ordre (sur les mesures brutes : le lissage ne biaise pas tau)
    #    vitesse : montée depuis 0 ; courant : décroissance à partir du pic
    fit_w = fit_i = None
    res_w = res_i = None
    if tourne and not sature:
        res_w = ajuster(t_w, w_b_nat, w_ss, 0.0)
        if res_w:
            fit_w = (t_w, modele(t_w, *res_w[:3]))
    k_pic = int(np.argmax(i_f))
    if tourne and len(t) - k_pic > 20:
        t_dec = t[k_pic:] - t[k_pic]
        res_i = ajuster(t_dec, i_b[k_pic:], i_ss, float(i_f[k_pic]))
        if res_i:
            fit_i = (t[k_pic:], modele(t_dec, *res_i[:3]))

    return {
        "fichier": nom,
        "pwm_nominal": pwm_nominal(nom, float(df["pwm_pct"].iloc[0])),
        "pwm_reel": float(df["pwm_pct"].iloc[0]),
        "n_points": int(len(t)),
        "te_ms": te * 1e3,
        "t": t,
        "courant_brut": i_b,
        "courant_filtre": i_f,
        "vitesse_brute": w_b,
        "vitesse_filtree": w_f,
        "courant_ss": i_ss,
        "courant_ondulation_eff": ond_eff,
        "courant_pic_filtre": float(i_f[k_pic]),
        "tau_courant": res_i[2] if res_i else np.nan,
        "r2_courant": res_i[3] if res_i else np.nan,
        "vitesse_ss": w_ss,
        "tau_vitesse": res_w[2] if res_w else np.nan,
        "r2_vitesse": res_w[3] if res_w else np.nan,
        "tourne": bool(tourne),
        "sature": sature,
        "regime_etabli": etabli,
        "utilisable": utilisable,
        "t_regime": t_regime,
        "fit_i": fit_i,
        "fit_w": fit_w,
    }


# ----------------------------Exports----------------------------- #
def ecrire_csv_filtre(r, dossier):
    os.makedirs(dossier, exist_ok=True)
    chemin = os.path.join(dossier, "filtre_" + r["fichier"])
    pd.DataFrame(
        {
            "temps_s": r["t"],
            "pwm_pct": r["pwm_nominal"],
            "courant_brut_A": r["courant_brut"],
            "courant_filtre_A": r["courant_filtre"],
            "vitesse_brute_rad_s": r["vitesse_brute"],
            "vitesse_filtree_rad_s": r["vitesse_filtree"],
        }
    ).to_csv(chemin, index=False, float_format="%.6f")
    return chemin


def ecrire_resume(resultats, dossier):
    os.makedirs(dossier, exist_ok=True)
    colonnes = [
        "fichier", "pwm_nominal", "pwm_reel", "n_points", "te_ms",
        "courant_ss", "courant_ondulation_eff", "courant_pic_filtre", "tau_courant", "r2_courant",
        "vitesse_ss", "tau_vitesse", "r2_vitesse",
        "tourne", "sature", "regime_etabli", "utilisable",
    ]  # fmt: skip
    df = pd.DataFrame([{c: r[c] for c in colonnes} for r in resultats])
    df = df.rename(
        columns={
            "pwm_nominal": "pwm_nominal_pct",
            "pwm_reel": "pwm_reel_pct",
            "te_ms": "Te_ms",
            "courant_ss": "courant_ss_A",
            "courant_ondulation_eff": "courant_ondulation_eff_A",
            "courant_pic_filtre": "courant_pic_filtre_A",
            "tau_courant": "tau_courant_s",
            "vitesse_ss": "vitesse_ss_rad_s",
            "tau_vitesse": "tau_vitesse_s",
        }
    ).sort_values(["pwm_nominal_pct", "fichier"])
    chemin = os.path.join(dossier, "resume_campagne.csv")
    df.to_csv(chemin, index=False, float_format="%.6g")
    return chemin, df


def ecrire_npz(resultats, dossier):
    os.makedirs(dossier, exist_ok=True)
    n = len(resultats)
    nmax = max(len(r["t"]) for r in resultats)

    def empile(cle):
        tab = np.full((n, nmax), np.nan)
        for k, r in enumerate(resultats):
            tab[k, : len(r[cle])] = r[cle]
        return tab

    chemin = os.path.join(dossier, "dataset_campagne.npz")
    np.savez(
        chemin,
        fichiers=np.array([r["fichier"] for r in resultats]),
        pwm_pct=np.array([r["pwm_nominal"] for r in resultats]),
        t=empile("t"),
        courant_brut=empile("courant_brut"),
        courant_filtre=empile("courant_filtre"),
        vitesse_brute=empile("vitesse_brute"),
        vitesse_filtree=empile("vitesse_filtree"),
        courant_ss=np.array([r["courant_ss"] for r in resultats]),
        vitesse_ss=np.array([r["vitesse_ss"] for r in resultats]),
        tau_vitesse=np.array([r["tau_vitesse"] for r in resultats]),
        utilisable=np.array([r["utilisable"] for r in resultats]),
    )
    return chemin


# ----------------------------Figures----------------------------- #
def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.6)
    for cote in ("top", "right"):
        ax.spines[cote].set_visible(False)
    for cote in ("left", "bottom"):
        ax.spines[cote].set_color(INK2)
    ax.tick_params(colors=INK2)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)
    ax.title.set_color(INK)


def tracer_essai(r, dossier):
    os.makedirs(dossier, exist_ok=True)
    t = r["t"]
    fig, (ax_i, ax_w) = plt.subplots(2, 1, sharex=True, figsize=(9, 6.5), facecolor=SURFACE, layout="constrained")
    for ax in (ax_i, ax_w):
        ax.axvspan(r["t_regime"], t[-1], color=FILTRE, alpha=0.06, lw=0)

    ax_i.plot(t, r["courant_brut"], color=BRUT, lw=0.8, label="mesure brute")
    ax_i.plot(t, r["courant_filtre"], color=FILTRE, lw=1.8, label="filtrée")
    if r["fit_i"] is not None:
        ax_i.plot(*r["fit_i"], color=AJUST, lw=1.4, ls="--", label=f"1er ordre (tau = {r['tau_courant'] * 1e3:.0f} ms)")
    ax_i.axhline(r["courant_ss"], color=INK2, lw=0.8, ls=":")
    ax_i.annotate(
        f"moyenne régime établi : {r['courant_ss'] * 1e3:.1f} mA",
        xy=(t[-1], r["courant_ss"]), xytext=(-6, 14), textcoords="offset points",
        ha="right", color=INK2, fontsize=9,
    )  # fmt: skip
    ax_i.set_ylabel("Courant (A)")
    ax_i.set_title(f"PWM {r['pwm_nominal']:g} %  -  {r['fichier']}", fontsize=10)
    ax_i.legend(frameon=False, loc="upper right", fontsize=9)

    ax_w.plot(t, r["vitesse_brute"], color=BRUT, lw=0.8, label="mesure brute")
    ax_w.plot(t, r["vitesse_filtree"], color=FILTRE, lw=1.8, label="filtrée")
    if r["fit_w"] is not None:
        t_fit, y_fit = r["fit_w"]
        ax_w.plot(t_fit, y_fit, color=AJUST, lw=1.4, ls="--", label=f"1er ordre (tau = {r['tau_vitesse'] * 1e3:.0f} ms)")
    if r["sature"]:
        ax_w.axhline(W_PLEIN_ECHELLE, color=INK2, lw=1.0, ls="--")
        ax_w.annotate(
            "plein échelle ADC (5 rad/s) : vitesse saturée, mesure inexploitable",
            xy=(t[len(t) // 2], W_PLEIN_ECHELLE), xytext=(0, -16), textcoords="offset points",
            ha="center", color=INK2, fontsize=9,
        )  # fmt: skip
        ax_w.set_ylim(-0.2, W_PLEIN_ECHELLE * 1.1)
    else:
        ax_w.annotate(
            f"moyenne régime établi : {r['vitesse_ss']:.3f} rad/s",
            xy=(t[-1], r["vitesse_ss"]), xytext=(-6, -16), textcoords="offset points",
            ha="right", color=INK2, fontsize=9,
        )  # fmt: skip
    ax_w.set_ylabel("Vitesse (rad/s)")
    ax_w.set_xlabel("Temps depuis l'échelon (s)")
    ax_w.legend(frameon=False, loc="center right", fontsize=9)
    for ax in (ax_i, ax_w):
        _style(ax)

    chemin = os.path.join(dossier, "essai_" + os.path.splitext(r["fichier"])[0] + ".png")
    fig.savefig(chemin, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    return chemin


def tracer_synthese(resultats, dossier):
    os.makedirs(dossier, exist_ok=True)
    rs = sorted(resultats, key=lambda r: r["pwm_nominal"])
    pwms = np.array([r["pwm_nominal"] for r in rs])
    vmax = pwms.max() if pwms.max() > pwms.min() else pwms.min() + 1
    norm = matplotlib.colors.Normalize(vmin=pwms.min(), vmax=vmax)
    rampe = matplotlib.colors.ListedColormap(plt.get_cmap("Blues")(np.linspace(0.3, 1.0, 256)))

    # Superposition des essais : une seule teinte, du clair (PWM faible) au foncé (PWM fort)
    fig, (a_i, a_w) = plt.subplots(2, 1, sharex=True, figsize=(9.5, 7), facecolor=SURFACE, layout="constrained")
    for r in rs:
        couleur = rampe(norm(r["pwm_nominal"]))
        a_i.plot(r["t"], r["courant_filtre"], color=couleur, lw=1.3)
        a_w.plot(r["t"], r["vitesse_filtree"], color=couleur, lw=1.3)
    a_w.axhline(W_PLEIN_ECHELLE, color=INK2, lw=0.9, ls="--")
    a_w.annotate("plein échelle ADC (5 rad/s)", xy=(0.02, W_PLEIN_ECHELLE), xytext=(0, -13),
                 textcoords="offset points", color=INK2, fontsize=9)  # fmt: skip
    a_i.set_ylabel("Courant filtré (A)")
    a_w.set_ylabel("Vitesse filtrée (rad/s)")
    a_w.set_xlabel("Temps depuis l'échelon (s)")
    a_i.set_title("Campagne d'essais : signaux filtrés, un tracé par niveau de PWM", fontsize=10)
    for ax in (a_i, a_w):
        _style(ax)
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=rampe), ax=[a_i, a_w], label="PWM (%)")
    cb.outline.set_edgecolor(INK2)
    fig.savefig(os.path.join(dossier, "synthese_signaux.png"), dpi=130, facecolor=SURFACE)
    plt.close(fig)

    # Grandeurs de régime établi en fonction du PWM
    ok = np.array([r["utilisable"] for r in rs])
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), facecolor=SURFACE, layout="constrained")
    specs = [
        ("vitesse_ss", "Vitesse en régime établi (rad/s)"),
        ("courant_ss", "Courant en régime établi (A)"),
        ("tau_vitesse", "Constante de temps de la vitesse (s)"),
    ]
    for ax, (cle, etiquette) in zip(axes, specs):
        y = np.array([r[cle] for r in rs], float)
        ax.plot(pwms[ok], y[ok], "o-", color=FILTRE, lw=1.5, ms=6, mec=SURFACE, mew=1.2, label="essai exploitable")
        if (~ok).any():
            ax.plot(pwms[~ok], y[~ok], "o", color=INK2, ms=7, mfc=SURFACE, mew=1.4,
                    label="non exploitable (saturé, non établi ou arrêté)")  # fmt: skip
        ax.set_xlabel("PWM (%)")
        ax.set_ylabel(etiquette)
        _style(ax)
    axes[0].legend(frameon=False, fontsize=8.5, loc="best")
    fig.savefig(os.path.join(dossier, "synthese_regime_etabli.png"), dpi=130, facecolor=SURFACE)
    plt.close(fig)


# -----------------------------Entrées---------------------------- #
def lister_csv(entrees):
    fichiers = []
    for e in entrees or [DOSSIER_ENTREE]:
        if os.path.isdir(e):
            fichiers += glob.glob(os.path.join(e, "*.csv"))
        else:
            fichiers += glob.glob(e)
    return sorted(set(fichiers))


# ----------------------------Lancement--------------------------- #
if __name__ == "__main__":
    parseur = argparse.ArgumentParser(description="Prétraitement de la campagne d'essais moteur CC")
    parseur.add_argument("entrees", nargs="*", help="dossier(s) ou fichier(s) CSV clean (défaut : csv/clean)")
    parseur.add_argument("--sortie", default=RACINE, help="dossier racine des sorties (défaut : à côté du script)")
    parseur.add_argument("--methode", choices=["butter", "savgol"], default=METHODE)
    parseur.add_argument("--fc", type=float, default=FC_HZ, help=f"coupure Butterworth en Hz (défaut : {FC_HZ:g})")
    parseur.add_argument("--ordre", type=int, default=ORDRE, help=f"ordre Butterworth (défaut : {ORDRE})")
    parseur.add_argument("--fenetre", type=int, default=FENETRE, help=f"fenêtre Savitzky-Golay en points (défaut : {FENETRE})")
    parseur.add_argument("--pre", type=float, default=T_PRE_S, help=f"temps de repos à supprimer en s (défaut : {T_PRE_S:g})")
    parseur.add_argument("--frac-regime", type=float, default=FRAC_REGIME, help="fraction finale = régime établi")
    parseur.add_argument("--sans-figures", action="store_true", help="ne pas générer les PNG")
    args = parseur.parse_args()

    fichiers = lister_csv(args.entrees)
    if not fichiers:
        raise SystemExit(f"Aucun CSV trouvé (dossier par défaut : {DOSSIER_ENTREE})")

    dossier_filtre = os.path.join(args.sortie, "csv", "filtre")
    dossier_dataset = os.path.join(args.sortie, "dataset")
    dossier_fig = os.path.join(args.sortie, "figures")

    print(f"{len(fichiers)} fichier(s), méthode = {args.methode}"
          + (f", fc = {args.fc:g} Hz, ordre {args.ordre}" if args.methode == "butter" else f", fenêtre = {args.fenetre} pts"))  # fmt: skip
    resultats = []
    for chemin in fichiers:
        print(f"- {os.path.basename(chemin)}")
        try:
            r = traiter_fichier(chemin, args.methode, args.fc, args.ordre, args.fenetre, args.pre, args.frac_regime)
            ecrire_csv_filtre(r, dossier_filtre)
            if not args.sans_figures:
                tracer_essai(r, dossier_fig)
            resultats.append(r)
        except Exception as erreur:
            print(f"  ERREUR : {erreur}. Fichier ignoré.")

    if not resultats:
        raise SystemExit("Aucun fichier traité.")

    chemin_resume, resume = ecrire_resume(resultats, dossier_dataset)
    chemin_npz = ecrire_npz(resultats, dossier_dataset)
    if not args.sans_figures:
        tracer_synthese(resultats, dossier_fig)

    print("\n" + resume[["pwm_nominal_pct", "courant_ss_A", "vitesse_ss_rad_s", "tau_vitesse_s",
                         "tau_courant_s", "sature", "regime_etabli", "utilisable"]].to_string(index=False))  # fmt: skip
    nb_ok = int(resume["utilisable"].sum())
    print(f"\n{nb_ok}/{len(resume)} essai(s) exploitable(s).")
    if (resume["sature"]).any():
        print("ATTENTION : vitesse saturée à 5 rad/s (plein échelle de l'ADC) pour : "
              + ", ".join(f"{p:g} %" for p in resume.loc[resume["sature"], "pwm_nominal_pct"]))  # fmt: skip
    print(f"CSV filtrés : {dossier_filtre}\nRésumé      : {chemin_resume}\nDataset     : {chemin_npz}")
    if not args.sans_figures:
        print(f"Figures     : {dossier_fig}")