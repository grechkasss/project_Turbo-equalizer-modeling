from chanel_MSA.chanel import *
bits = generate_bits(10000, seed=42)
channels = get_proakis_channels()
results_a = simulate_proakis(
    bits,
    channels["Proakis A"],
    snr_db=10,
    seed=42)
results_b = simulate_proakis(
    bits,
    channels["Proakis B"],
    snr_db=10,
    seed=42)
results_c = simulate_proakis(
    bits,
    channels["Proakis C"],
    snr_db=10,
    seed=42)
print("Proakis A BER:", results_a["ber"])
print("Proakis B BER:", results_b["ber"])
print("Proakis C BER:", results_c["ber"])

plot_signal(results_a)
plot_signal(results_b)
plot_signal(results_c)