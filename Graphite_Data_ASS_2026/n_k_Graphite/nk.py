import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 1. Read CSV files
# =========================
# Each file: col-1 = Energy (eV), col-2 = epsilon values

eps1_data = pd.read_csv("epsilon1.csv", header=1)
eps2_data = pd.read_csv("epsilon2.csv", header=1)

E1 = eps1_data.iloc[:, 0].values
eps1 = eps1_data.iloc[:, 1].values

E2 = eps2_data.iloc[:, 0].values
eps2 = eps2_data.iloc[:, 1].values

# =========================
# 2. Create common energy grid
# =========================
E_min = max(E1.min(), E2.min())
E_max = min(E1.max(), E2.max())

# choose resolution (adjust if needed)
E_common = np.linspace(E_min, E_max, 200)

# =========================
# 3. Interpolate eps1, eps2
# =========================
eps1_interp = np.interp(E_common, E1, eps1)
eps2_interp = np.interp(E_common, E2, eps2)

# =========================
# 4. Convert to n, k
# =========================
mod_eps = np.sqrt(eps1_interp**2 + eps2_interp**2)

n = np.sqrt((mod_eps + eps1_interp) / 2)
k = np.sqrt((mod_eps - eps1_interp) / 2)

# =========================
# 5. Convert to wavelength
# =========================
lambda_nm = 1240.0 / E_common

# =========================
# 6. Optional: sort by wavelength
# =========================
# (useful for plotting in optics convention)
sort_idx = np.argsort(lambda_nm)[::-1]

lambda_nm = lambda_nm[sort_idx]
n = n[sort_idx]
k = k[sort_idx]

# =========================
# 7. Plot
# =========================
plt.figure()
plt.rcParams["font.size"] = 14
plt.plot(lambda_nm, n, label='n',linestyle='-')
plt.plot(lambda_nm, k, label='k',linestyle='--')

plt.xlabel('Wavelength (nm)')
plt.ylabel('Optical Constants')
plt.title('n and k from dielectric data')
plt.legend()
plt.grid()

plt.show()

output = pd.DataFrame({
    "lambda_nm": lambda_nm,
    "n": n,
    "k": k
})

output.to_csv("nk_data.csv", index=False)