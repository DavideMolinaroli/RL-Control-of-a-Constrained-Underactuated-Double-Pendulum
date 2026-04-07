import numpy as np
import matplotlib.pyplot as plt

def penalty_th1(x):
    return 2 * np.log(-np.abs(x) + 4*np.pi)
def penalty_th2(x):
    return 2 * np.log(-np.abs(x) + 4*np.pi + np.pi/2)
def penalty_th3(x):
    return 2 * np.log(-np.abs(x) + 4*np.pi + np.pi/6)

# Define x range
x = np.linspace(-5*np.pi, 5*np.pi, 2000)

# Mask domain (inside log must be > 0)
mask = -np.abs(x) + 4*np.pi > 0
mask1 = -np.abs(x) + 4*np.pi + np.pi/2 > 0
mask2 = -np.abs(x) + 4*np.pi + np.pi/6 > 0
x_valid = x[mask]
x_valid1 = x[mask1]
x_valid2 = x[mask2]
y_valid = penalty_th1(x_valid)
y_valid1 = penalty_th2(x_valid1)
y_valid2 = penalty_th3(x_valid2)

# Plot
plt.figure(figsize=(6, 3))
plt.plot(x_valid, y_valid, color="b", linewidth=1, label="slack=0")
plt.plot(x_valid1, y_valid1, color="y", linewidth=1, label="slack=π/2")
plt.plot(x_valid2, y_valid2, color="g", linewidth=1, label="slack=π/6")

# X-axis ticks every π
xticks = np.arange(-5*np.pi, 5*np.pi + np.pi, np.pi)
xtick_labels = [f"{i}π" if i != 0 else "0" for i in range(-5, 6)]
plt.xticks(xticks, xtick_labels)
plt.axvline(x=4*np.pi, color="r", linestyle="--", linewidth=1)
plt.axvline(x=-4*np.pi, color="r", linestyle="--", linewidth=1)
plt.axvline(x=0, color="k", linestyle=":", linewidth=1)
plt.axhline(y=0, color="k", linestyle=":", linewidth=1)

plt.xlabel("ϑ [rad]")
plt.ylabel("penalty(ϑ)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()

plt.tight_layout()
plt.savefig("penalty_th1.pdf")
plt.close()
