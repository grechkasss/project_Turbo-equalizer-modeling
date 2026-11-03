import numpy as np
import matplotlib.pyplot as plt

def generate_bits(n_bits, seed=None):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, n_bits)

def bpsk_modulate(bits):
    return 2 * bits - 1

def bpsk_demodulate(signal):
    return (signal.real >= 0).astype(int)

def get_proakis_channels():
    return {
        "Proakis A": np.array([0.26, 0.93, 0.26]),
        "Proakis B": np.array([0.407, 0.815, 0.407]),
        "Proakis C": np.array([0.227, 0.46, 0.688, 0.46, 0.227])
    }

def apply_proakis_channel(signal, channel):
    return np.convolve(signal, channel, mode="full")

def add_awgn(signal, snr_db, seed=None):
    rng = np.random.default_rng(seed)

    signal_power = np.mean(np.abs(signal) ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear

    noise = np.sqrt(noise_power / 2) * (
        rng.normal(size=signal.shape)
        + 1j * rng.normal(size=signal.shape)
    )

    return signal + noise

def normalize_powers(power_db):
    power_linear = 10 ** (np.asarray(power_db) / 10)
    return power_linear / np.sum(power_linear)

def scale_delays(normalized_delays, rms_delay_spread):
    normalized_delays = np.asarray(normalized_delays)
    return normalized_delays * rms_delay_spread

def delays_to_samples(delays, sampling_rate):
    return np.rint(delays * sampling_rate).astype(int)

def generate_jakes_fading(
    n_samples,
    sampling_rate,
    max_doppler,
    seed=None,
    n_sinusoids=64
):
    rng = np.random.default_rng(seed)
    time = np.arange(n_samples) / sampling_rate
    angles = 2 * np.pi * rng.random(n_sinusoids)
    phases = 2 * np.pi * rng.random(n_sinusoids)
    doppler_frequencies = max_doppler * np.cos(angles)
    fading = np.sum(
        np.exp(
            1j * (
                2 * np.pi
                * doppler_frequencies[:, None]
                * time[None, :]
                + phases[:, None]
            )
        ),
        axis=0
    )
    fading = fading / np.sqrt(n_sinusoids)
    fading_power = np.mean(np.abs(fading) ** 2)
    if fading_power > 0:
        fading = fading / np.sqrt(fading_power)

    return fading


def generate_rayleigh_fading(
    n_samples,
    sampling_rate,
    max_doppler,
    seed=None
):
    rng = np.random.default_rng(seed)

    fading_i = generate_jakes_fading(
        n_samples,
        sampling_rate,
        max_doppler,
        seed
    )
    fading_q = generate_jakes_fading(
        n_samples,
        sampling_rate,
        max_doppler,
        seed + 1 if seed is not None else None
    )
    fading = (fading_i + 1j * fading_q) / np.sqrt(2)
    fading_power = np.mean(np.abs(fading) ** 2)
    if fading_power > 0:
        fading = fading / np.sqrt(fading_power)

    return fading


def generate_rician_fading(
    n_samples,
    sampling_rate,
    max_doppler,
    k_factor_db,
    los_angle=0.0,
    seed=None
):
    k_factor = 10 ** (k_factor_db / 10)
    diffuse = generate_rayleigh_fading(
        n_samples,
        sampling_rate,
        max_doppler,
        seed
    )
    time = np.arange(n_samples) / sampling_rate
    los = np.exp(
        1j * (
            2 * np.pi * max_doppler
            * np.cos(los_angle)
            * time
        )
    )
    fading = (
        np.sqrt(k_factor / (k_factor + 1)) * los
        + np.sqrt(1 / (k_factor + 1)) * diffuse
    )
    return fading


def apply_tdl_channel(
    signal,
    delays,
    powers_db,
    sampling_rate,
    fading_type="rayleigh",
    max_doppler=0.0,
    k_factor_db=6.0,
    los_angle=0.0,
    seed=None
):
    delays = np.asarray(delays, dtype=float)
    powers_db = np.asarray(powers_db, dtype=float)
    if len(delays) != len(powers_db):
        raise ValueError("Delays and powers must have equal lengths")
    if np.any(delays < 0):
        raise ValueError("Delays must be non-negative")
    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be positive")
    if max_doppler < 0:
        raise ValueError("Maximum Doppler must be non-negative")
    power_linear = normalize_powers(powers_db)
    sample_delays = delays_to_samples(
        delays,
        sampling_rate
    )

    n_samples = len(signal) + np.max(sample_delays)
    output = np.zeros(n_samples, dtype=complex)
    rng = np.random.default_rng(seed)
    for m in range(len(delays)):
        delay = sample_delays[m]
        amplitude = np.sqrt(power_linear[m])
        if fading_type == "rayleigh":
            fading = generate_rayleigh_fading(
                n_samples,
                sampling_rate,
                max_doppler,
                seed + m if seed is not None else None
            )
        elif fading_type == "rician":
            if m == 0:
                fading = generate_rician_fading(
                    n_samples,
                    sampling_rate,
                    max_doppler,
                    k_factor_db,
                    los_angle,
                    seed
                )
            else:
                fading = generate_rayleigh_fading(
                    n_samples,
                    sampling_rate,
                    max_doppler,
                    seed + m if seed is not None else None
            )
        elif fading_type == "static":
            phase = rng.uniform(0, 2 * np.pi)
            fading = np.full(
                n_samples,
                np.exp(1j * phase),
                dtype=complex
            )
        else:
            raise ValueError(
                "fading_type must be rayleigh, rician or static"
            )

        delayed_signal = np.zeros(n_samples, dtype=complex)
        delayed_signal[delay:delay + len(signal)] = signal
        output += amplitude * fading * delayed_signal
    return output

def simulate_proakis(bits, channel, snr_db, seed=None):
    symbols = bpsk_modulate(bits)
    channel_output = apply_proakis_channel(
        symbols,
        channel
    )
    received_signal = add_awgn(
        channel_output,
        snr_db,
        seed
    )
    detected_bits = bpsk_demodulate(
        received_signal[:len(bits)]
    )
    ber = np.mean(bits != detected_bits)
    return {
        "symbols": symbols,
        "channel_output": channel_output,
        "received_signal": received_signal,
        "detected_bits": detected_bits,
        "ber": ber
    }


def simulate_tdl(
    bits,
    delays,
    powers_db,
    sampling_rate,
    snr_db,
    fading_type="rayleigh",
    max_doppler=0.0,
    k_factor_db=6.0,
    los_angle=0.0,
    seed=None
):
    symbols = bpsk_modulate(bits)

    channel_output = apply_tdl_channel(
        symbols,
        delays,
        powers_db,
        sampling_rate,
        fading_type,
        max_doppler,
        k_factor_db,
        los_angle,
        seed
    )

    received_signal = add_awgn(
        channel_output,
        snr_db,
        seed
    )

    detected_bits = bpsk_demodulate(
        received_signal[:len(bits)]
    )

    ber = np.mean(bits != detected_bits)

    return {
        "symbols": symbols,
        "channel_output": channel_output,
        "received_signal": received_signal,
        "detected_bits": detected_bits,
        "ber": ber
    }


def plot_signal(results, n_samples=100):
    symbols = results["symbols"]
    channel_output = results["channel_output"]
    received_signal = results["received_signal"]

    n = min(n_samples, len(symbols))

    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(symbols[:n].real, marker="o")
    plt.title("Переданный BPSK-сигнал")
    plt.grid()

    plt.subplot(3, 1, 2)
    plt.plot(channel_output[:n].real, marker="o")
    plt.title("Сигнал после канала")
    plt.grid()

    plt.subplot(3, 1, 3)
    plt.plot(received_signal[:n].real, marker="o")
    plt.title("Сигнал после AWGN")
    plt.grid()

    plt.tight_layout()
    plt.show()


def calculate_ber_snr(
    bits,
    channel_function,
    channel_parameters,
    snr_values,
    seed=42
):
    ber_values = []

    for snr_db in snr_values:
        results = channel_function(
            bits=bits,
            snr_db=snr_db,
            seed=seed,
            **channel_parameters
        )
        ber_values.append(results["ber"])
    return np.array(ber_values)

def plot_ber(snr_values, ber_results):
    plt.figure(figsize=(10, 6))
    for name, ber_values in ber_results.items():
        plt.semilogy(
            snr_values,
            ber_values,
            marker="o",
            label=name
        )
    plt.xlabel("SNR, дБ")
    plt.ylabel("BER")
    plt.title("Сравнение каналов")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()
