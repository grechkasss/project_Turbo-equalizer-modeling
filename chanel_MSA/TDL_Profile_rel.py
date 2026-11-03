from chanel_MSA.chanel import *

tdl_profiles = {
    "TDL-A": {
        "delays": np.array([0, 30, 70, 100, 150]) * 1e-9,
        "powers_db": np.array([0, -3, -6, -8, -10]),
        "fading_type": "rayleigh"
    },

    "TDL-B": {
        "delays": np.array([0, 50, 100, 200, 300]) * 1e-9,
        "powers_db": np.array([0, -2, -5, -8, -12]),
        "fading_type": "rayleigh"
    },

    "TDL-C": {
        "delays": np.array([0, 10, 20, 40]) * 1e-9,
        "powers_db": np.array([0, -4, -8, -12]),
        "fading_type": "rayleigh"
    },

    "TDL-D": {
        "delays": np.array([0, 30, 70, 100, 150]) * 1e-9,
        "powers_db": np.array([0, -3, -6, -8, -10]),
        "fading_type": "rician"
    },

    "TDL-E": {
        "delays": np.array([0, 10, 20, 40]) * 1e-9,
        "powers_db": np.array([0, -4, -8, -12]),
        "fading_type": "rician"
    }
}
bits = generate_bits(10000, seed=42)
profile = tdl_profiles["TDL-D"]
results_tdl_d = simulate_tdl(
    bits=bits,
    delays=profile["delays"],
    powers_db=profile["powers_db"],
    sampling_rate=100e6,
    snr_db=10,
    fading_type=profile["fading_type"],
    max_doppler=100,
    k_factor_db=6,
    los_angle=0,
    seed=42
)
print("TDL-D BER:", results_tdl_d["ber"])
plot_signal(results_tdl_d)