# ------------------------------------------------------------ #
# Fichier : train_modele.py
# Date : 07/11/26
# Description : modèle de commande du moteur CC, consigne de vitesse -> PWM (%)
#   1) modèle STATIQUE (boucle ouverte) : vitesse établie -> PWM
#   2) modèle INVERSE DYNAMIQUE (prépare la boucle fermée) :
#      (vitesse mesurée, vitesse visée dans H * dt) -> PWM
#   Chaque réseau est comparé à une référence simple (droite, 1er ordre) et validé
#   en retirant un NIVEAU de PWM entier (leave-one-level-out) : les essais répétés
#   d'un même niveau ne sont jamais à la fois dans l'entraînement et dans le test.
#   Seuls les essais marqués "utilisable" par preprocess_moteur.py sont utilisés.
# ------------------------------------------------------------ #

# ------------------------------------------------------------ #
# Lancement :
#   python train_modele.py                          (dataset/dataset_campagne.npz à côté du script)
#   python train_modele.py chemin/vers/fichier.npz  --horizon 5 --dt 0.01
# Sorties (dossier modele/ à côté du script) :
#   modele_regulateur.joblib    modèles entraînés sur tous les essais exploitables
#   metriques_validation.csv    erreurs de validation en points de PWM (%)
#   validation_modele.png       figure de validation
# Utilisation en boucle ouverte :
#   import joblib
#   m = joblib.load("modele/modele_regulateur.joblib")
#   pwm = pwm_pour_vitesse(m, 3.0)        # consigne 3 rad/s -> PWM en %
#   (la fonction pwm_pour_vitesse est définie dans ce fichier)
# ------------------------------------------------------------ #

# -------------------------Librairie---------------------------- #
import argparse
import os
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ---------------------------Paramètres--------------------------- #
RACINE = os.path.dirname(os.path.abspath(__file__))
NPZ_DEFAUT = os.path.join(RACINE, "dataset", "dataset_campagne.npz")
DOSSIER_SORTIE = os.path.join(RACINE, "modele")

DT_S = 0.01  # pas de rééchantillonnage des séries pour le modèle dynamique (s)
HORIZON = 5  # vitesse visée dans HORIZON * DT_S secondes (50 ms ~ 1 tau)
T_MAX_S = 0.5  # on garde le début de l'essai (accélération), le reste est du régime établi
T_TRANSITOIRE_S = 0.25  # avant cet instant : transitoire ; après : régime établi
TAU_MIN_PWM = 45  # le tau du 1er ordre est pris sur les niveaux >= 45 % (moteur lancé)
GRAINE = 0

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLEU, ORANGE = "#2a78d6", "#eb6834"


# ---------------------------Données------------------------------ #
def charger(chemin):
    d = np.load(chemin)
    idx = np.flatnonzero(d["utilisable"])
    if len(idx) < 6:
        raise SystemExit(f"Seulement {len(idx)} essai(s) exploitable(s) : trop peu pour entraîner.")
    essais = []
    for k in idx:
        n = int(np.isfinite(d["t"][k]).sum())
        essais.append(
            {
                "fichier": str(d["fichiers"][k]),
                "pwm": float(d["pwm_pct"][k]),
                "w_ss": float(d["vitesse_ss"][k]),
                "tau": float(d["tau_vitesse"][k]),
                "t": d["t"][k][:n],
                "w": d["vitesse_filtree"][k][:n],
            }
        )
    return essais, int(len(d["utilisable"]))


def construire_paires(essais, dt, h, t_max):
    """Paires (w_k, w_{k+h}) -> PWM appliqué, le long de chaque essai."""
    lignes = []
    tg = np.arange(0.0, t_max + 1e-9, dt)
    for j, e in enumerate(essais):
        wg = np.interp(tg, e["t"], e["w"])
        for s in range(len(tg) - h):
            lignes.append((wg[s], wg[s + h], e["pwm"], tg[s], j))
    a = np.array(lignes)
    return a[:, :2], a[:, 2], a[:, 3], a[:, 4].astype(int)


# ---------------------------Modèles------------------------------ #
def mlp(couches, graine=GRAINE):
    return TransformedTargetRegressor(
        regressor=make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=couches, activation="tanh", solver="lbfgs", max_iter=5000, random_state=graine),
        ),
        transformer=StandardScaler(),
    )


def physique_ajuster(wss, pwm, tau, dt, h):
    """Inverse d'un 1er ordre : w_ss estimée = (w_{k+h} - e^-a * w_k) / (1 - e^-a),
    puis PWM = droite(w_ss). Deux paramètres appris : la droite ; tau est mesuré."""
    pente, ordonnee = np.polyfit(wss, pwm, 1)
    masque = pwm >= TAU_MIN_PWM
    tau_med = float(np.median(tau[masque] if masque.any() else tau))
    return {"pente": float(pente), "ordonnee": float(ordonnee), "tau": tau_med, "dt": dt, "h": h}


def physique_predire(p, w_k, w_kh):
    e = np.exp(-p["h"] * p["dt"] / p["tau"])
    return p["pente"] * ((w_kh - e * w_k) / (1 - e)) + p["ordonnee"]


def pwm_pour_vitesse(modeles, w_ref):
    """Boucle ouverte : PWM (%) pour une vitesse établie visée (rad/s).
    La consigne est bornée à la plage réellement mesurée ; le PWM à [0, 100]."""
    w_min, w_max = modeles["plage_vitesse"]
    w = float(np.clip(w_ref, w_min, w_max))
    return float(np.clip(modeles["statique_droite"].predict([[w]])[0], 0, 100))


# --------------------------Validation---------------------------- #
def rmse(e):
    return float(np.sqrt(np.mean(np.square(e)))) if len(e) else float("nan")


def lolo_statique(fabrique, wss, pwm):
    pred = np.empty_like(pwm)
    for niveau in np.unique(pwm):
        te = pwm == niveau
        pred[te] = fabrique().fit(wss[~te, None], pwm[~te]).predict(wss[te, None])
    return pred


def lolo_dynamique(xb, yb, jb, pwm_essai, wss, tau, niveaux, dt, h, graine):
    pred_p, pred_m = np.zeros(len(yb)), np.zeros(len(yb))
    pwm_b = pwm_essai[jb]
    for niveau in niveaux:
        te = pwm_b == niveau
        pred_m[te] = mlp((16, 16), graine).fit(xb[~te], yb[~te]).predict(xb[te])
        tr = pwm_essai != niveau
        p = physique_ajuster(wss[tr], pwm_essai[tr], tau[tr], dt, h)
        pred_p[te] = physique_predire(p, xb[te, 0], xb[te, 1])
    return pred_p, pred_m


# ----------------------------Figure------------------------------ #
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


def figure_validation(chemin, wss, pwm, niveaux, droite, reseau, err_dr, err_ml, plage, essais, donnees_dyn, pred_p, pred_m):
    xb, yb, tk, jb = donnees_dyn
    fig = plt.figure(figsize=(13, 8), facecolor=SURFACE, layout="constrained")
    gs = fig.add_gridspec(2, 3)

    ax = fig.add_subplot(gs[0, 0:2])
    w_plot = np.linspace(plage[0] - 0.3, plage[1] + 0.3, 200)[:, None]
    ax.scatter(wss, pwm, s=38, color=INK2, edgecolor=SURFACE, zorder=3, label="essais exploitables")
    ax.plot(w_plot, droite.predict(w_plot), color=BLEU, lw=1.8, label="droite")
    ax.plot(w_plot, reseau.predict(w_plot), color=ORANGE, lw=1.8, ls="--", label="réseau (MLP)")
    for borne in plage:
        ax.axvline(borne, color=INK2, lw=0.8, ls=":")
    ax.annotate("limites des données mesurées", xy=(plage[1], pwm.min()), xytext=(-6, 6), textcoords="offset points",
                ha="right", color=INK2, fontsize=9)  # fmt: skip
    ax.set_xlabel("Vitesse établie visée (rad/s)")
    ax.set_ylabel("PWM à appliquer (%)")
    ax.set_title("Modèle statique : vitesse -> PWM", fontsize=10)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    _style(ax)

    ax = fig.add_subplot(gs[0, 2])
    ax.axhline(0, color=INK2, lw=0.8)
    ax.plot(niveaux, [err_dr[pwm == n].mean() for n in niveaux], "o", color=BLEU, ms=7, mec=SURFACE, label="droite")
    ax.plot(niveaux, [err_ml[pwm == n].mean() for n in niveaux], "D", color=ORANGE, ms=6, mec=SURFACE, label="réseau (MLP)")
    ax.set_xlabel("Niveau de PWM retiré de l'entraînement (%)")
    ax.set_ylabel("Erreur de PWM prédit (points de %)")
    ax.set_title("Validation : un niveau retiré à la fois", fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    _style(ax)

    interieurs = [n for n in niveaux if niveaux.min() < n < niveaux.max()]
    choix = [interieurs[1], interieurs[len(interieurs) // 2], interieurs[-2]]
    for c, niveau in enumerate(choix):
        ax = fig.add_subplot(gs[1, c])
        ax.axhline(niveau, color=INK2, lw=1.0, ls=":", label="PWM réellement appliqué")
        for j, e in enumerate(essais):
            if e["pwm"] != niveau:
                continue
            m = jb == j
            ax.plot(tk[m], pred_p[m], color=BLEU, lw=1.6, label="1er ordre + droite" if j == min(k for k, x in enumerate(essais) if x["pwm"] == niveau) else None)
            ax.plot(tk[m], pred_m[m], color=ORANGE, lw=1.6, ls="--", label="réseau (MLP)" if j == min(k for k, x in enumerate(essais) if x["pwm"] == niveau) else None)
        ax.set_ylim(niveau - 12, niveau + 12)
        ax.set_xlabel("Temps depuis l'échelon (s)")
        ax.set_ylabel("PWM prédit (%)")
        ax.set_title(f"Modèle dynamique, niveau {niveau:g} % retiré", fontsize=10)
        if c == 0:
            ax.legend(frameon=False, fontsize=8.5, loc="upper right")
        _style(ax)

    fig.savefig(chemin, dpi=130, facecolor=SURFACE)
    plt.close(fig)


# ----------------------------Lancement--------------------------- #
if __name__ == "__main__":
    parseur = argparse.ArgumentParser(description="Entraînement du modèle de commande (consigne de vitesse -> PWM)")
    parseur.add_argument("npz", nargs="?", default=NPZ_DEFAUT, help="dataset_campagne.npz (défaut : dataset/ à côté du script)")
    parseur.add_argument("--sortie", default=DOSSIER_SORTIE, help="dossier des sorties (défaut : modele/ à côté du script)")
    parseur.add_argument("--dt", type=float, default=DT_S, help=f"pas du modèle dynamique en s (défaut : {DT_S:g})")
    parseur.add_argument("--horizon", type=int, default=HORIZON, help=f"horizon en pas (défaut : {HORIZON})")
    parseur.add_argument("--graine", type=int, default=GRAINE)
    args = parseur.parse_args()

    essais, n_total = charger(args.npz)
    pwm = np.array([e["pwm"] for e in essais])
    wss = np.array([e["w_ss"] for e in essais])
    tau = np.array([e["tau"] for e in essais])
    niveaux = np.unique(pwm)
    interieur = (pwm > niveaux.min()) & (pwm < niveaux.max())
    plage = (float(wss.min()), float(wss.max()))
    print(f"{len(essais)}/{n_total} essais exploitables, {len(niveaux)} niveaux de PWM ({niveaux.min():g} à {niveaux.max():g} %)")
    print(f"Plage de vitesse couverte : {plage[0]:.2f} à {plage[1]:.2f} rad/s")

    # ---- 1) modèle statique -------------------------------------------------
    fabriques = {
        "droite": LinearRegression,
        "mlp": lambda: mlp((8, 8), args.graine),
    }
    err = {nom: lolo_statique(f, wss, pwm) - pwm for nom, f in fabriques.items()}
    lignes = []
    for nom, e in err.items():
        lignes.append({"modele": f"statique_{nom}", "partie": "niveaux intérieurs",
                    "rmse_pwm": rmse(e[interieur]), "n": int(interieur.sum())})  # fmt: skip
        lignes.append({"modele": f"statique_{nom}", "partie": "niveaux extrêmes (extrapolation)",
                    "rmse_pwm": rmse(e[~interieur]), "n": int((~interieur).sum())})  # fmt: skip

    # ---- 2) modèle inverse dynamique -------------------------------------------
    xb, yb, tk, jb = construire_paires(essais, args.dt, args.horizon, T_MAX_S)
    inter_b = interieur[jb]
    pred_p, pred_m = lolo_dynamique(xb, yb, jb, pwm, wss, tau, niveaux, args.dt, args.horizon, args.graine)
    for nom, p in (("dynamique_1er_ordre", pred_p), ("dynamique_mlp", pred_m)):
        e = p - yb
        lignes.append({"modele": nom, "partie": "transitoire, niveaux intérieurs",
                    "rmse_pwm": rmse(e[inter_b & (tk <= T_TRANSITOIRE_S)]), "n": int((inter_b & (tk <= T_TRANSITOIRE_S)).sum())})  # fmt: skip
        lignes.append({"modele": nom, "partie": "régime établi, niveaux intérieurs",
                    "rmse_pwm": rmse(e[inter_b & (tk > T_TRANSITOIRE_S)]), "n": int((inter_b & (tk > T_TRANSITOIRE_S)).sum())})  # fmt: skip

    os.makedirs(args.sortie, exist_ok=True)
    metriques = pd.DataFrame(lignes)
    metriques.to_csv(os.path.join(args.sortie, "metriques_validation.csv"), index=False, float_format="%.3f")
    print("\nErreur de validation (RMSE, en points de PWM, niveau retiré de l'entraînement) :")
    print(metriques.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    # répétabilité des mesures : plancher d'erreur que ne peut pas battre un modèle
    pente = np.polyfit(wss, pwm, 1)[0]
    ecarts = [np.ptp(wss[pwm == n]) * abs(pente) for n in niveaux if (pwm == n).sum() > 1]
    if ecarts:
        print(f"\nRépétabilité entre essais d'un même niveau : {np.mean(ecarts):.2f} point de PWM en moyenne "
            f"(max {np.max(ecarts):.2f}) : plancher d'erreur des modèles.")  # fmt: skip

    # ---- 3) entraînement final sur tous les essais exploitables ---------------------
    droite = LinearRegression().fit(wss[:, None], pwm)
    reseau = mlp((8, 8), args.graine).fit(wss[:, None], pwm)
    dyn_mlp = mlp((16, 16), args.graine).fit(xb, yb)
    dyn_phys = physique_ajuster(wss, pwm, tau, args.dt, args.horizon)
    modeles = {
        "statique_droite": droite,
        "statique_mlp": reseau,
        "dynamique_mlp": dyn_mlp,
        "dynamique_1er_ordre": dyn_phys,
        "plage_vitesse": plage,
        "niveaux_pwm": niveaux,
        "parametres": {"dt": args.dt, "horizon": args.horizon, "graine": args.graine},
    }
    joblib.dump(modeles, os.path.join(args.sortie, "modele_regulateur.joblib"))
    figure_validation(os.path.join(args.sortie, "validation_modele.png"), wss, pwm, niveaux, droite, reseau,
                    err["droite"], err["mlp"], plage, essais, (xb, yb, tk, jb), pred_p, pred_m)  # fmt: skip

    print(f"\nModèle statique retenu : PWM [%] = {droite.coef_[0]:.3f} * vitesse [rad/s] {droite.intercept_:+.3f}"
        f"   (valable de {plage[0]:.2f} à {plage[1]:.2f} rad/s)")  # fmt: skip
    print(f"Modèle dynamique 1er ordre : tau = {dyn_phys['tau'] * 1e3:.0f} ms, horizon = {args.horizon * args.dt * 1e3:.0f} ms")
    print(f"Sorties : {args.sortie}")