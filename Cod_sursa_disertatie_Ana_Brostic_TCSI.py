import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import tensorflow as tf
from keras import layers, Model, callbacks
import os, warnings, joblib
warnings.filterwarnings("ignore")

# reproductibilitatea rezultatelor
np.random.seed(42)
tf.random.set_seed(42)

# configurare parametrii
DATA_DIR        = r"D:\Master 1\Cercetare\data"
FEATURES        = ["Flow Duration", "Total Fwd Packets", "Total Backward Packets",
                   "Fwd Packet Length Mean", "Bwd Packet Length Mean",
                   "Flow IAT Mean", "Destination Port"]
WINDOW          = 100       # fereastra glisanta entropie Shannon
STEP            = 50        # pas pentru fereastra (overlap 50%)
LATENT_DIM      = 4         # dimensiunea stratului latent (bottleneck)
EPOCHS          = 100       # numar maxim de epoci pentru antrenare
BATCH           = 256       # dimensiune batch pentru antrenare
SIGMA_MULT      = 0.5       # multiplicatorul lui sigma pentru pragul adaptiv
OUT_DIR         = r"D:\Master 1\Cercetare\output"
os.makedirs(OUT_DIR, exist_ok=True)

# 1. DATE
def load_data():
    csvs = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")] if os.path.isdir(DATA_DIR) else []
    if csvs:
        frames = []
        for f in csvs:
            header = pd.read_csv(os.path.join(DATA_DIR, f), nrows=0)
            stripped_to_orig = {c.strip(): c for c in header.columns}
            label_col_orig = next((c for c in header.columns if "label" in c.strip().lower()), None)
            missing = [feat for feat in FEATURES if feat not in stripped_to_orig]
            if missing:
                print(f"[AVERT] {f}: features lipsa {missing}")

            needed_orig = [stripped_to_orig[feat] for feat in FEATURES if feat in stripped_to_orig]
            if label_col_orig:
                needed_orig.append(label_col_orig)

            # incarcare in ordinea naturala a fisierului necesara pentru calculul corect al entropiei Shannon pe ferestre glisante
            chunk = pd.read_csv(
                os.path.join(DATA_DIR, f),
                usecols=needed_orig,
                dtype={c: np.float32 for c in needed_orig if c != label_col_orig},
                low_memory=True,
            )
            chunk.columns = chunk.columns.str.strip()
            frames.append(chunk)

        df = pd.concat(frames, ignore_index=True)
        label_col = next((c for c in df.columns if "label" in c.lower()), None)
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        df["Label"] = (df[label_col].str.upper() != "BENIGN").astype(int)

    else:
        print("Date CIC-DDoS2019 negasite, se genereaza date sintetice")
        rng = np.random.default_rng(42)
        n_b, n_a = 8000, 4000
        benign = pd.DataFrame({
            "Flow Duration":           rng.exponential(1e6, n_b),
            "Total Fwd Packets":       rng.poisson(20, n_b),
            "Total Backward Packets":  rng.poisson(15, n_b),
            "Fwd Packet Length Mean":  rng.normal(500, 100, n_b).clip(0),
            "Bwd Packet Length Mean":  rng.normal(300, 80,  n_b).clip(0),
            "Flow IAT Mean":           rng.exponential(5000, n_b),
            "Destination Port":        rng.integers(1024, 65535, n_b),
            "Label": 0
        })
        attack = pd.DataFrame({
            "Flow Duration":           rng.exponential(2e5, n_a),
            "Total Fwd Packets":       rng.poisson(50,  n_a),
            "Total Backward Packets":  rng.poisson(2,   n_a),
            "Fwd Packet Length Mean":  rng.normal(100, 20,  n_a).clip(0),
            "Bwd Packet Length Mean":  rng.normal(50,  10,  n_a).clip(0),
            "Flow IAT Mean":           rng.normal(200, 30,  n_a).clip(0),
            "Destination Port":        rng.choice([80, 443, 8080], n_a),
            "Label": 1
        })
        df = pd.concat([benign, attack], ignore_index=True).sample(frac=1, random_state=42)
    return df

# 2. ENTROPIE SHANNON
def shannon_entropy(series):
    vals, counts = np.unique(series, return_counts=True)
    p = counts / counts.sum()
    # Formula entropiei Shannon; 1e-10 previne log2(0) care ar da -inf
    return -np.sum(p * np.log2(p + 1e-10))

def add_entropy_features(df):
    df = df.reset_index(drop=True)
    for col in ["Flow Duration", "Total Fwd Packets", "Destination Port"]:
        if col not in df.columns:
            continue
        ent = []
        total = len(range(0, len(df) - WINDOW, STEP))
        for idx_i, i in enumerate(range(0, len(df) - WINDOW, STEP)):
            if idx_i % 10000 == 0:
                print(f"entropie {col}: {idx_i}/{total} ferestre procesate", end="\r")
            ent.append(shannon_entropy(df[col].iloc[i:i+WINDOW].values))
        print(f"entropie {col}: {total}/{total} ferestre procesate")
        idx = np.linspace(0, len(df)-1, len(ent)).astype(int)
        col_full = np.full(len(df), np.nan)
        col_full[idx] = ent
        # .bfill() dupa .interpolate() elimina NaN-urile de la inceput
        df[f"Entropy_{col.replace(' ', '_')}"] = pd.Series(col_full).interpolate().bfill().values
    return df

# 3. PREPROCESARE
def preprocess(df):
    df = add_entropy_features(df)
    feat = [c for c in FEATURES + [f for f in df.columns if f.startswith("Entropy")] if c in df.columns]
    df = df[feat + ["Label"]].dropna()

    labels = df["Label"].values
    X = df[feat].values

    benign_idx = np.where(labels == 0)[0]
    np.random.shuffle(benign_idx)
    split = int(0.8 * len(benign_idx))
    X_train = X[benign_idx[:split]]
    X_val   = X[benign_idx[split:]]

    attack_idx = np.where(labels == 1)[0]
    np.random.shuffle(attack_idx)
    n_test = min(10_000, len(benign_idx[split:]), len(attack_idx))
    test_idx = np.concatenate([benign_idx[split:split + n_test], attack_idx[:n_test]])
    X_test   = X[test_idx]
    y_test   = labels[test_idx]

    # scaler antrenat doar pe train benign pentru a evita scurgerea de date 
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train) # invata MIN si MAX din train, apoi aplica
    X_val   = scaler.transform(X_val) # aplica acelasi MIN si MAX invatat pe validare
    X_test  = scaler.transform(X_test) # aplica acelasi MIN si MAX invatat pe test

    return X_train, X_val, X_test, y_test, feat, scaler

# 4. MODEL
def build_autoencoder(input_dim):
    # assert garanteaza compresie reala, bottleneck < input
    assert LATENT_DIM < input_dim, \
        f"bottleneck ({LATENT_DIM}) trebuie < input_dim ({input_dim}) pentru compresie reala"

    x = inp = layers.Input(shape=(input_dim,), name="x_input")
    for units in [128, 64, 32]:
        x = layers.Dense(units, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
    bottleneck = layers.Dense(LATENT_DIM, activation="relu", name="bottleneck")(x)
    x = bottleneck
    for units in [32, 64, 128]:
        x = layers.Dense(units, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
    out = layers.Dense(input_dim, activation="sigmoid", name="z_reconstruction")(x)
    model = Model(inp, out, name="DeepAutoencoder")
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model

# 5. ANTRENARE
def train(model, X_train, X_val):
    cb = [
        callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(patience=5, factor=0.5),
    ]
    return model.fit(X_train, X_train, validation_data=(X_val, X_val),
                     epochs=EPOCHS, batch_size=BATCH, callbacks=cb, verbose=1)

# 6. PRAG ADAPTIV
def compute_threshold(model, X_val):
    recon = model.predict(X_val, verbose=0)
    mse = np.mean((X_val - recon) ** 2, axis=1)
    theta = mse.mean() + SIGMA_MULT * mse.std()
    print(f"Θ = {mse.mean():.6f} + {SIGMA_MULT}·{mse.std():.6f} = {theta:.6f}")
    return theta


# 7. EVALUARE
def evaluate(model, X_test, y_test, theta):
    recon = model.predict(X_test, verbose=0)
    mse   = np.mean((X_test - recon) ** 2, axis=1)
    y_pred = (mse > theta).astype(int)
    # orice flux cu MSE>Θ este considerat atac

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2,2) else (0,0,0,0)
    prec   = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1     = 2 * prec * recall / (prec + recall + 1e-9)
    fpr    = fp / (fp + tn + 1e-9)
    auc    = roc_auc_score(y_test, mse)

    print(f"\n  Precizie={prec:.4f}  Sensibilitate={recall:.4f}  F1={f1:.4f}  AUC={auc:.4f}  FPR={fpr:.4f}")
    print(classification_report(y_test, y_pred, target_names=["BENIGN", "ATAC"]))
    return mse, y_pred, {"Precizie": prec, "Sensibilitate": recall, "F1": f1, "AUC": auc, "FPR": fpr}

# 8. GRAFICE
def plot_all(history, mse, y_test, y_pred, theta, metrics):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(history.history["loss"], label="Eroare antrenare")
    ax.plot(history.history["val_loss"], label="Eroare validare")
    ax.set_title("Evoluția erorii de antrenare și validare")
    ax.legend(); ax.set_xlabel("Epocă"); ax.set_ylabel("Eroare MSE")
    plt.tight_layout()
    path1 = os.path.join(OUT_DIR, "plot_erori.png")
    plt.savefig(path1, dpi=150); plt.close()
    print(f"Grafic salvat: {path1}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(mse[y_test==0], bins=60, alpha=0.6, label="Benign", color="steelblue")
    ax.hist(mse[y_test==1], bins=60, alpha=0.6, label="Atac", color="tomato")
    ax.axvline(theta, color="k", linestyle="--", label=f"Θ={theta:.4f}")
    ax.set_title("Distribuția erorii de reconstrucție MSE")
    ax.legend(); ax.set_xlabel("Eroare de reconstrucție MSE"); ax.set_ylabel("Număr fluxuri")
    plt.tight_layout()
    path2 = os.path.join(OUT_DIR, "plot_mse_distributie.png")
    plt.savefig(path2, dpi=150); plt.close()
    print(f"Grafic salvat: {path2}")

    fig, ax = plt.subplots(figsize=(5, 5))
    cm = confusion_matrix(y_test, y_pred)
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i,j], ha="center", va="center", fontsize=14)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Benign","Atac"]); ax.set_yticklabels(["Benign","Atac"])
    ax.set_xlabel("Predicție"); ax.set_ylabel("Real")
    ax.set_title("Matricea de confuzie")
    plt.tight_layout()
    path3 = os.path.join(OUT_DIR, "plot_confuzie.png")
    plt.savefig(path3, dpi=150); plt.close()
    print(f"Grafic salvat: {path3}")

    fig, ax = plt.subplots(figsize=(7, 5))
    keys = ["Precizie","Sensibilitate","F1","AUC"]
    ax.bar(keys, [metrics[k] for k in keys], color=["steelblue","tomato","seagreen","darkorange"])
    ax.set_ylim(0, 1.1)
    for i, k in enumerate(keys):
        ax.text(i, metrics[k] + 0.02, f"{metrics[k]:.3f}", ha="center", fontsize=11)
    ax.set_title("Metrici de performanță")
    plt.tight_layout()
    path4 = os.path.join(OUT_DIR, "plot_metrici.png")
    plt.savefig(path4, dpi=150); plt.close()
    print(f"Grafic salvat: {path4}")

# MAIN
if __name__ == "__main__":
    print("=" * 60)
    print("LDoS Detection - Autoencoder + Shannon Entropy")
    print("=" * 60)

    df = load_data()
    X_train, X_val, X_test, y_test, features, scaler = preprocess(df)
    print(f"Features: {len(features)}  |  Train: {len(X_train)}  |  Test: {len(X_test)}")

    model = build_autoencoder(X_train.shape[1])
    model.summary()

    history = train(model, X_train, X_val)
    theta   = compute_threshold(model, X_val)
    mse, y_pred, metrics = evaluate(model, X_test, y_test, theta)
    plot_all(history, mse, y_test, y_pred, theta, metrics)

    model.save(os.path.join(OUT_DIR, "autoencoder.keras"))
    print("Model salvat output/autoencoder.keras")

    joblib.dump(scaler, os.path.join(OUT_DIR, "scaler.pkl"))
    print("Scaler salvat output/scaler.pkl")

    print("=" * 60)