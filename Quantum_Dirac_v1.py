# Paul Dirac 


"""
Quantum_Dirac_V1.py

Python 3.11
Qiskit 1.4.4

DIRACOV MODEL
-------------
    i dPsi/dt = H Psi

    H = c * sigma_x tensor P
        + mu * sigma_z tensor I
        + I tensor V

    mu = m*c^2

Koriste se jedinice hbar=1 i vreme evolucije t=1.

P je Hermitski spektralni operator impulsa na periodicnoj
prostornoj mrezi sa 39 brojnih stanja.

V je nauceni realni potencijal sa cetiri prostorne komponente.

Za slobodan model:
    H^2 = c^2 * I tensor P^2 + mu^2 * I.

KUBITI
------
Sedam zasebnih registara.
Svaki ima sest prostornih i jedan spinorski kubit.

Qiskit aktivni indeksi:
    spinor 0: 0..38
    spinor 1: 64..102

Ostala bazna stanja su neaktivna.
Gust vektor od 49 kubita se ne pravi.

UCENJE
------
Istorija odredjuje pocetni spinor.
Prethodni broj iste sortirane pozicije odredjuje
pocetno prostorno stanje.

L-BFGS-B uci parametre kvantnog modela.
Gubitak je zajednicki uslovni NLL legalnih kombinacija.
Gradijent je analiticki, ukljucujuci Frechetov adjoint.

L2 kazna obuhvata istorijske koeficijente i potencijal.
Osnovni spinorski uglovi, c i mu nemaju L2 kaznu.

Hronoloska validacija, odvojeni test i zavrsno ucenje
na svim istorijskim prelazima za svaki CSV zasebno.

ISPRAVKA QISKIT PROVERE
----------------------
Qiskit i numericki simulator porede se preko amplituda
i Bornovih verovatnoca, uz proveru normalizacije.

Ne porede se neposredno logaritmi veoma malih verovatnoca.
Logaritmi se i dalje koriste za NLL i globalni MAP.

Nema slucajnog generisanja kombinacija.
Nema zaokruzivanja predikcija i popravljanja duplikata.
Nema kozmetickih filtera.
"""

import csv
import hashlib
import itertools
import random
from pathlib import Path

import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Statevector, partial_trace
from scipy.linalg import expm
from scipy.optimize import minimize
from scipy.special import logsumexp


random.seed(39)
np.random.seed(39)


FILES = (
    Path("/Users/4c/Desktop/GHQ/data/loto7_4716_k89.csv"),
    Path("/Users/4c/Desktop/GHQ/data/loto7_4716_k89_loto_2980.csv"),
    Path("/Users/4c/Desktop/GHQ/data/loto7_4716_k89_loto_plus_1736.csv"),
)

N = 39
K = 7
POSITION_QUBITS = 6
POSITION_DIM = 1 << POSITION_QUBITS
REGISTER_QUBITS = POSITION_QUBITS + 1
REGISTER_DIM = 1 << REGISTER_QUBITS
ACTIVE_DIM = 2 * N

WINDOWS = (1, 3)
REGULARIZATIONS = (0.001, 0.01)

TRAIN_FRACTION = 0.70
VALIDATION_END_FRACTION = 0.85

BATCH_SIZE = 128
MAX_ITERATIONS = 200
GRADIENT_TOLERANCE = 1e-5
RESTARTS = 2
REPORT_EVERY = 10

SPEED_BOUNDS = (0.25, 20.0)
MASS_ENERGY_BOUNDS = (0.05, 15.0)
POTENTIAL_BOUNDS = (-8.0, 8.0)

ACTIVE_INDICES = np.concatenate([
    np.arange(N),
    POSITION_DIM + np.arange(N),
])

SIGMA_X = np.array([
    [0.0, 1.0],
    [1.0, 0.0],
], dtype=np.complex128)

SIGMA_Z = np.array([
    [1.0, 0.0],
    [0.0, -1.0],
], dtype=np.complex128)

IDENTITY_SPIN = np.eye(2, dtype=np.complex128)
IDENTITY_POSITION = np.eye(N, dtype=np.complex128)

GRID = np.arange(N, dtype=np.float64)

FOURIER = np.exp(
    -2j * np.pi * np.outer(GRID, GRID) / N
) / np.sqrt(N)

MOMENTA = 2.0 * np.pi * np.fft.fftfreq(N)

MOMENTUM = (
    FOURIER.conj().T
    @ (MOMENTA[:, None] * FOURIER)
)
MOMENTUM = 0.5 * (MOMENTUM + MOMENTUM.conj().T)

POTENTIAL_BASIS = np.stack([
    np.sin(2.0 * np.pi * GRID / N),
    np.cos(2.0 * np.pi * GRID / N),
    np.sin(4.0 * np.pi * GRID / N),
    np.cos(4.0 * np.pi * GRID / N),
])

GENERATORS = np.stack(
    [
        np.kron(SIGMA_X, MOMENTUM),
        np.kron(SIGMA_Z, IDENTITY_POSITION),
    ]
    + [
        np.kron(IDENTITY_SPIN, np.diag(column))
        for column in POTENTIAL_BASIS
    ],
    axis=0,
)


class DistributionError(ArithmeticError):
    pass


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def load_csv(path):
    raw = path.read_bytes()
    rows = []

    for line_number, row in enumerate(
        csv.reader(raw.decode("utf-8-sig").splitlines()),
        start=1,
    ):
        if not row or all(not cell.strip() for cell in row):
            continue

        require(
            len(row) == K,
            f"{path.name}, red {line_number}: potrebno je sedam kolona.",
        )

        try:
            values = [int(cell.strip()) for cell in row]
        except ValueError as exc:
            raise RuntimeError(
                f"{path.name}, red {line_number}: necelobrojna vrednost."
            ) from exc

        require(
            len(set(values)) == K
            and all(1 <= value <= N for value in values)
            and values == sorted(values),
            f"{path.name}, red {line_number}: nelegalna kombinacija.",
        )
        rows.append(values)

    require(len(rows) >= 5, f"{path.name}: nedovoljno kola.")

    return {
        "path": path,
        "data": np.asarray(rows, dtype=np.int64) - 1,
        "digest": hashlib.sha256(raw).hexdigest(),
    }


def raw_history_features(data, times, window):
    times = np.asarray(times, dtype=np.int64)

    require(
        window in WINDOWS
        and times.ndim == 1
        and len(times) > 0
        and np.all((times >= 1) & (times <= len(data))),
        "Neispravna istorijska vremena ili prozor.",
    )

    result = np.zeros(
        (len(times), K * window + window - 1),
        dtype=np.float64,
    )

    for lag in range(1, window + 1):
        source = times - lag
        available = source >= 0
        start = K * (lag - 1)

        result[available, start:start + K] = (
            data[source[available]].astype(float) + 1.0 - 20.0
        ) / 19.0

        if lag >= 2:
            result[:, K * window + lag - 2] = available.astype(float)

    return result


def fit_transform(raw, training_rows):
    require(
        1 <= training_rows <= len(raw),
        "Neispravna granica skaliranja.",
    )

    training = raw[:training_rows]
    mean = training.mean(axis=0)
    scale = training.std(axis=0)
    keep = scale > 1e-12

    return {
        "mean": mean,
        "scale": scale,
        "keep": keep,
    }


def transform_features(raw, transform):
    keep = transform["keep"]

    result = np.ones(
        (len(raw), 1 + int(np.sum(keep))),
        dtype=np.float64,
    )

    if np.any(keep):
        result[:, 1:] = (
            raw[:, keep] - transform["mean"][keep]
        ) / transform["scale"][keep]

    require(np.isfinite(result).all(), "Nekonacne karakteristike.")
    return result


def parameter_count(feature_count):
    return 2 * feature_count + len(GENERATORS)


def parameter_bounds(feature_count):
    one_register = (
        [(None, None)] * (2 * feature_count)
        + [SPEED_BOUNDS, MASS_ENERGY_BOUNDS]
        + [POTENTIAL_BOUNDS] * 4
    )
    return one_register * K


def regularization_mask(feature_count):
    mask = np.zeros(
        (K, parameter_count(feature_count)),
        dtype=np.float64,
    )

    mask[:, 1:feature_count] = 1.0
    mask[:, feature_count + 1:2 * feature_count] = 1.0
    mask[:, 2 * feature_count + 2:] = 1.0

    return mask.ravel()


def initial_parameters(feature_count, restart):
    parameters = np.zeros(
        (K, parameter_count(feature_count)),
        dtype=np.float64,
    )
    indices = np.arange(feature_count, dtype=np.float64)

    for slot in range(K):
        phase = (
            indices + 1.0 + 0.31 * slot + 0.73 * restart
        )

        parameters[slot, :feature_count] = (
            0.04 * np.sin(phase) / np.sqrt(feature_count)
        )
        parameters[slot, feature_count:2 * feature_count] = (
            0.04 * np.cos(phase) / np.sqrt(feature_count)
        )

        parameters[slot, 0] = (
            np.pi / 2.0 + 0.12 * np.sin(slot + 1.0)
        )
        parameters[slot, feature_count] = (
            0.40 + 0.10 * np.cos(slot + 1.0)
        )

        parameters[slot, 2 * feature_count] = (
            7.0 + 0.35 * slot + 5.0 * restart
        )
        parameters[slot, 2 * feature_count + 1] = (
            1.8 + 0.10 * slot + 1.2 * restart
        )
        parameters[slot, 2 * feature_count + 2:] = (
            0.15 * np.sin(
                np.arange(1, 5, dtype=float)
                * (slot + 1.0)
                + 0.3 * restart
            )
        )

    return parameters.ravel()


def make_operator(coefficients, with_adjoint=False):
    coefficients = np.asarray(coefficients, dtype=np.float64)

    require(
        coefficients.shape == (len(GENERATORS),)
        and np.isfinite(coefficients).all(),
        "Neispravni Diracovi koeficijenti.",
    )

    hamiltonian = np.tensordot(
        coefficients, GENERATORS, axes=(0, 0)
    )

    require(
        np.allclose(
            hamiltonian,
            hamiltonian.conj().T,
            atol=2e-11,
            rtol=0.0,
        ),
        "Diracov Hamiltonijan nije Hermitski.",
    )

    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    phases = np.exp(-1j * eigenvalues)

    unitary = (
        eigenvectors * phases[None, :]
    ) @ eigenvectors.conj().T

    frechet = None

    if with_adjoint:
        difference = (
            eigenvalues[:, None] - eigenvalues[None, :]
        )
        average = (
            eigenvalues[:, None] + eigenvalues[None, :]
        ) / 2.0

        frechet = (
            -1j
            * np.exp(-1j * average)
            * np.sinc(difference / (2.0 * np.pi))
        )

    return {
        "hamiltonian": hamiltonian,
        "unitary": unitary,
        "vectors": eigenvectors,
        "frechet": frechet,
    }


def operator_adjoint(operator, unitary_adjoint):
    vectors = operator["vectors"]
    frechet = operator["frechet"]

    require(frechet is not None, "Nedostaje Frechetov izvod.")

    transformed = (
        vectors.conj().T @ unitary_adjoint @ vectors
    )
    hamiltonian_adjoint = (
        vectors
        @ (frechet.conj() * transformed)
        @ vectors.conj().T
    )

    hamiltonian_adjoint = 0.5 * (
        hamiltonian_adjoint + hamiltonian_adjoint.conj().T
    )

    return np.real(np.einsum(
        "ab,pab->p",
        hamiltonian_adjoint.conj(),
        GENERATORS,
    ))


def forward_register(theta, features, previous, operator):
    feature_count = features.shape[1]
    previous = np.asarray(previous, dtype=np.int64)

    require(
        previous.shape == (len(features),)
        and np.all((previous >= 0) & (previous < N)),
        "Neispravna pocetna prostorna stanja.",
    )

    angle = features @ theta[:feature_count]
    phase = features @ theta[feature_count:2 * feature_count]

    exp_minus = np.exp(-0.5j * phase)
    exp_plus = np.exp(0.5j * phase)

    spin_zero = np.cos(angle / 2.0) * exp_minus
    spin_one = np.sin(angle / 2.0) * exp_plus

    columns_zero = operator["unitary"][:, previous].T
    columns_one = operator["unitary"][:, N + previous].T

    values = (
        spin_zero[:, None] * columns_zero
        + spin_one[:, None] * columns_one
    )

    require(
        np.isfinite(values).all()
        and np.allclose(
            np.sum(np.abs(values) ** 2, axis=1),
            1.0,
            atol=2e-9,
            rtol=0.0,
        ),
        "Diracov spinor nije normalizovan.",
    )

    return values, {
        "angle": angle,
        "exp_minus": exp_minus,
        "exp_plus": exp_plus,
        "spin_zero": spin_zero,
        "spin_one": spin_one,
        "columns_zero": columns_zero,
        "columns_one": columns_one,
    }


def log_amplitudes(amplitudes):
    absolute = np.abs(amplitudes)
    result = np.full(absolute.shape, -np.inf)
    positive = absolute > 0.0
    result[positive] = 2.0 * np.log(absolute[positive])
    return result


def position_logs(values):
    spinors = values.reshape(len(values), 2, N)
    logs = logsumexp(log_amplitudes(spinors), axis=1)

    require(
        np.allclose(
            logsumexp(logs, axis=1),
            0.0,
            atol=2e-9,
            rtol=0.0,
        ),
        "Prostorna Bornova raspodela nije normalizovana.",
    )
    return logs


def conditional_statistics(log_weights, targets=None):
    logs = np.asarray(log_weights, dtype=np.float64).copy()

    require(
        logs.ndim == 3
        and not np.isnan(logs).any()
        and not np.isposinf(logs).any(),
        "Neispravna zajednicka raspodela.",
    )

    batch, slots, states = logs.shape

    require(
        batch > 0 and 1 <= slots <= states,
        "Neispravne dimenzije raspodele.",
    )

    for slot in range(slots):
        logs[:, slot, :slot] = -np.inf
        logs[:, slot, states - slots + slot + 1:] = -np.inf

    forward = np.full_like(logs, -np.inf)
    backward = np.full_like(logs, -np.inf)
    forward[:, 0] = logs[:, 0]

    for slot in range(1, slots):
        prefix = np.logaddexp.accumulate(
            forward[:, slot - 1], axis=1
        )
        strict_prefix = np.concatenate([
            np.full((batch, 1), -np.inf),
            prefix[:, :-1],
        ], axis=1)
        forward[:, slot] = logs[:, slot] + strict_prefix

    log_z = logsumexp(forward[:, -1], axis=1)

    if not np.isfinite(log_z).all():
        raise DistributionError("Nema pozitivne legalne mase.")

    backward[:, -1] = 0.0

    for slot in range(slots - 2, -1, -1):
        next_values = logs[:, slot + 1] + backward[:, slot + 1]
        suffix = np.logaddexp.accumulate(
            next_values[:, ::-1], axis=1
        )[:, ::-1]

        backward[:, slot] = np.concatenate([
            suffix[:, 1:],
            np.full((batch, 1), -np.inf),
        ], axis=1)

    marginals = np.exp(
        forward + backward - log_z[:, None, None]
    )

    require(
        np.all(log_z <= 2e-8)
        and np.allclose(
            marginals.sum(axis=2),
            1.0,
            atol=2e-8,
            rtol=0.0,
        ),
        "Legalna raspodela nije normalizovana.",
    )

    losses = None

    if targets is not None:
        targets = np.asarray(targets, dtype=np.int64)

        require(
            targets.shape == (batch, slots)
            and np.all((targets >= 0) & (targets < states))
            and np.all(np.diff(targets, axis=1) > 0),
            "Neispravni ciljevi.",
        )

        target_logs = logs[
            np.arange(batch)[:, None],
            np.arange(slots)[None, :],
            targets,
        ].sum(axis=1)

        losses = log_z - target_logs

        require(
            not np.isnan(losses).any()
            and np.all(losses >= -2e-8),
            "Neispravan uslovni NLL.",
        )
        losses = np.maximum(losses, 0.0)

    return {
        "logs": logs,
        "log_z": log_z,
        "marginals": marginals,
        "losses": losses,
    }


def joint_distribution(log_weights):
    statistics = conditional_statistics(
        np.asarray(log_weights)[None, :, :]
    )
    logs = statistics["logs"][0]
    slots, states = logs.shape

    maximum = logs[0].copy()
    parents = np.full((slots, states), -1, dtype=np.int64)
    increasing = (
        np.arange(states)[:, None]
        < np.arange(states)[None, :]
    )

    for slot in range(1, slots):
        candidates = np.where(
            increasing, maximum[:, None], -np.inf
        )
        parents[slot] = candidates.argmax(axis=0)
        maximum = logs[slot] + candidates[
            parents[slot], np.arange(states)
        ]

    path = np.empty(slots, dtype=np.int64)
    path[-1] = int(np.argmax(maximum))

    for slot in range(slots - 1, 0, -1):
        path[slot - 1] = parents[slot, path[slot]]

    log_z = float(statistics["log_z"][0])
    log_mode = float(logs[np.arange(slots), path].sum())

    require(
        np.all(np.diff(path) > 0)
        and np.isfinite(log_mode)
        and log_mode <= log_z + 2e-8,
        "Neispravan globalni MAP.",
    )

    return {
        "path": path,
        "logs": logs,
        "log_z": log_z,
        "marginals": statistics["marginals"][0],
        "log_mode_probability": log_mode - log_z,
    }


def spinor_adjoint(values, log_probability_gradient):
    spinors = values.reshape(len(values), 2, N)
    magnitude = np.max(np.abs(spinors), axis=1)

    scaled = np.zeros_like(spinors)
    np.divide(
        spinors,
        magnitude[:, None, :],
        out=scaled,
        where=magnitude[:, None, :] > 0.0,
    )

    scaled_norm = np.sum(np.abs(scaled) ** 2, axis=1)
    denominator = magnitude * scaled_norm
    result = np.zeros_like(spinors)

    with np.errstate(
        divide="ignore", invalid="ignore", over="ignore"
    ):
        np.divide(
            2.0 * log_probability_gradient[:, None, :] * scaled,
            denominator[:, None, :],
            out=result,
            where=denominator[:, None, :] > 0.0,
        )

    return result.reshape(len(values), ACTIVE_DIM)


def objective(
    theta, features, previous, targets, weight,
    with_gradient=True,
):
    total = len(targets)
    feature_count = features.shape[1]
    count = parameter_count(feature_count)

    require(
        total > 0
        and len(features) == total
        and previous.shape == targets.shape == (total, K),
        "Neispravan trening skup.",
    )

    parameters = np.asarray(theta).reshape(K, count)
    operators = [
        make_operator(
            parameters[slot, 2 * feature_count:],
            with_adjoint=with_gradient,
        )
        for slot in range(K)
    ]

    gradient = np.zeros_like(parameters)
    unitary_adjoints = [
        np.zeros((ACTIVE_DIM, ACTIVE_DIM), dtype=np.complex128)
        for _ in range(K)
    ]
    loss_sum = 0.0

    for start in range(0, total, BATCH_SIZE):
        stop = min(start + BATCH_SIZE, total)
        batch_features = features[start:stop]
        batch_previous = previous[start:stop]
        batch_targets = targets[start:stop]

        states = []
        caches = []

        for slot in range(K):
            state, cache = forward_register(
                parameters[slot],
                batch_features,
                batch_previous[:, slot],
                operators[slot],
            )
            states.append(state)
            caches.append(cache)

        logs = np.stack(
            [position_logs(state) for state in states],
            axis=1,
        )

        try:
            statistics = conditional_statistics(
                logs, batch_targets
            )
        except DistributionError:
            return np.inf, np.zeros_like(theta)

        if not np.isfinite(statistics["losses"]).all():
            return np.inf, np.zeros_like(theta)

        loss_sum += float(statistics["losses"].sum())

        if not with_gradient:
            continue

        log_gradient = statistics["marginals"].copy()
        log_gradient[
            np.arange(stop - start)[:, None],
            np.arange(K)[None, :],
            batch_targets,
        ] -= 1.0

        for slot in range(K):
            adjoint = spinor_adjoint(
                states[slot], log_gradient[:, slot]
            )

            if not np.isfinite(adjoint).all():
                return np.inf, np.zeros_like(theta)

            cache = caches[slot]
            angle = cache["angle"]

            derivative_zero = (
                -0.5
                * np.sin(angle / 2.0)
                * cache["exp_minus"]
            )
            derivative_one = (
                0.5
                * np.cos(angle / 2.0)
                * cache["exp_plus"]
            )

            angle_derivative = (
                derivative_zero[:, None] * cache["columns_zero"]
                + derivative_one[:, None] * cache["columns_one"]
            )
            phase_derivative = (
                -0.5j
                * cache["spin_zero"][:, None]
                * cache["columns_zero"]
                + 0.5j
                * cache["spin_one"][:, None]
                * cache["columns_one"]
            )

            angle_gradient = np.real(np.sum(
                adjoint.conj() * angle_derivative,
                axis=1,
            ))
            phase_gradient = np.real(np.sum(
                adjoint.conj() * phase_derivative,
                axis=1,
            ))

            gradient[slot, :feature_count] += (
                angle_gradient @ batch_features
            )
            gradient[
                slot, feature_count:2 * feature_count
            ] += phase_gradient @ batch_features

            np.add.at(
                unitary_adjoints[slot].T,
                batch_previous[:, slot],
                adjoint * cache["spin_zero"].conj()[:, None],
            )
            np.add.at(
                unitary_adjoints[slot].T,
                N + batch_previous[:, slot],
                adjoint * cache["spin_one"].conj()[:, None],
            )

    if with_gradient:
        for slot in range(K):
            gradient[slot, 2 * feature_count:] = operator_adjoint(
                operators[slot], unitary_adjoints[slot]
            )

    mask = regularization_mask(feature_count)
    denominator = max(1.0, float(mask.sum()))

    penalty = weight * np.dot(mask * theta, theta) / denominator
    penalty_gradient = (
        2.0 * weight * mask * theta / denominator
    )

    loss = loss_sum / total + penalty
    full_gradient = gradient.ravel() / total + penalty_gradient

    require(
        np.isfinite(loss) and np.isfinite(full_gradient).all(),
        "Nekonacan gubitak ili gradijent.",
    )

    return float(loss), full_gradient


def projected_gradient(theta, gradient, bounds):
    result = gradient.copy()

    for index, (lower, upper) in enumerate(bounds):
        if (
            lower is not None
            and theta[index] <= lower + 1e-8
            and gradient[index] > 0.0
        ):
            result[index] = 0.0

        if (
            upper is not None
            and theta[index] >= upper - 1e-8
            and gradient[index] < 0.0
        ):
            result[index] = 0.0

    return result


def fit_models(features, previous, targets, weight, label):
    print(label, flush=True)
    print(f"Trening prelaza: {len(targets)}")
    print(f"Broj karakteristika sa interceptom: {features.shape[1]}")
    print(f"L2: {weight}")

    feature_count = features.shape[1]
    bounds = parameter_bounds(feature_count)
    best = None

    for restart in range(RESTARTS):
        tracked = {
            "loss": np.inf,
            "theta": None,
            "calls": 0,
        }

        def tracked_objective(candidate):
            value, gradient = objective(
                candidate, features, previous, targets, weight
            )
            tracked["calls"] += 1

            if np.isfinite(value) and value < tracked["loss"]:
                tracked["loss"] = value
                tracked["theta"] = candidate.copy()

            if tracked["calls"] % REPORT_EVERY == 0:
                print(
                    f"Procena {tracked['calls']}, "
                    f"najbolji NLL+L2={tracked['loss']:.9f}",
                    flush=True,
                )

            return value, gradient

        print(
            f"Deterministicki pocetak {restart + 1}/{RESTARTS}",
            flush=True,
        )

        result = minimize(
            tracked_objective,
            initial_parameters(feature_count, restart),
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={
                "maxiter": MAX_ITERATIONS,
                "gtol": GRADIENT_TOLERANCE,
                "ftol": 1e-10,
                "maxls": 30,
                "maxcor": 10,
            },
        )

        require(
            tracked["theta"] is not None,
            "Optimizator nije nasao konacan kandidat.",
        )

        chosen = tracked["theta"]
        loss, gradient = objective(
            chosen, features, previous, targets, weight
        )
        projected = projected_gradient(chosen, gradient, bounds)
        gradient_norm = float(np.max(np.abs(projected)))

        print(f"Sacuvani NLL+L2: {loss:.9f}")
        print(f"Max|projektovani gradijent|: {gradient_norm:.3e}")
        print(f"Status: {result.message}")

        if gradient_norm > GRADIENT_TOLERANCE:
            print("Napomena: mali projektovani gradijent nije potvrdjen.")

        if best is None or loss < best["loss"]:
            best = {
                "loss": loss,
                "theta": chosen.copy(),
            }

    return best["theta"].reshape(
        K, parameter_count(feature_count)
    )


def predict_logs(models, features, previous, operators=None):
    feature_count = features.shape[1]

    if operators is None:
        operators = [
            make_operator(models[slot, 2 * feature_count:])
            for slot in range(K)
        ]

    logs = np.empty((len(features), K, N), dtype=np.float64)

    for start in range(0, len(features), BATCH_SIZE):
        stop = min(start + BATCH_SIZE, len(features))

        for slot in range(K):
            state, _ = forward_register(
                models[slot],
                features[start:stop],
                previous[start:stop, slot],
                operators[slot],
            )
            logs[start:stop, slot] = position_logs(state)

    return logs


def evaluate(models, features, previous, targets):
    require(len(targets) > 0, "Prazan evaluacioni skup.")

    logs = predict_logs(models, features, previous)
    statistics = conditional_statistics(logs, targets)

    hits = []

    for row, target in enumerate(targets):
        joint = joint_distribution(logs[row])
        hits.append(
            len(set(joint["path"].tolist()) & set(target.tolist()))
        )

    losses = statistics["losses"]

    return {
        "nll": float(losses.mean()),
        "hits": np.asarray(hits, dtype=np.int64),
        "zero_targets": int(np.isposinf(losses).sum()),
    }


def print_metrics(label, metrics):
    print(label)
    print(f"Srednji uslovni NLL: {metrics['nll']:.9f}")
    print(f"Srednji broj pogodaka: {metrics['hits'].mean():.6f}")
    print(f"Ciljevi sa nultom masom: {metrics['zero_targets']}")
    print(
        "Broj kola po pogocima 0..7: "
        + str(np.bincount(metrics["hits"], minlength=K + 1).tolist())
    )


def verify_qiskit(theta, feature, previous):
    feature = np.asarray(feature).reshape(1, -1)
    feature_count = feature.shape[1]

    operator = make_operator(theta[2 * feature_count:])
    full_unitary = np.eye(REGISTER_DIM, dtype=np.complex128)
    full_unitary[np.ix_(ACTIVE_INDICES, ACTIVE_INDICES)] = (
        operator["unitary"]
    )

    gate = UnitaryGate(full_unitary, label="Dirac_evolution")

    angle = float(feature[0] @ theta[:feature_count])
    phase = float(
        feature[0] @ theta[feature_count:2 * feature_count]
    )

    circuit = QuantumCircuit(REGISTER_QUBITS)

    for bit in range(POSITION_QUBITS):
        if (int(previous) >> bit) & 1:
            circuit.x(bit)

    circuit.ry(angle, POSITION_QUBITS)
    circuit.rz(phase, POSITION_QUBITS)
    circuit.append(gate, range(REGISTER_QUBITS))

    state = Statevector.from_instruction(circuit)
    actual = np.asarray(state.data)

    expected, _ = forward_register(
        theta, feature, np.array([previous]), operator
    )

    require(
        np.allclose(
            actual[ACTIVE_INDICES],
            expected[0],
            atol=2e-9,
            rtol=2e-9,
        ),
        "Qiskit i Diracov simulator daju razlicite amplitude.",
    )

    inactive = np.ones(REGISTER_DIM, dtype=bool)
    inactive[ACTIVE_INDICES] = False

    require(
        np.allclose(actual[inactive], 0.0, atol=2e-12),
        "Masa je presla u neaktivna prostorna stanja.",
    )

    upper = actual[:N]
    lower = actual[POSITION_DIM:POSITION_DIM + N]

    logs = logsumexp(
        log_amplitudes(np.stack([upper, lower])),
        axis=0,
    )

    # ISPRAVKA:
    # Porede se Bornove verovatnoce i normalizacija.
    # Direktno poređenje logaritama malih masa nije stabilno.
    actual_probabilities = np.abs(upper) ** 2 + np.abs(lower) ** 2
    expected_probabilities = np.sum(
        np.abs(expected[0].reshape(2, N)) ** 2,
        axis=0,
    )

    require(
        np.allclose(
            actual_probabilities,
            expected_probabilities,
            atol=2e-10,
            rtol=2e-8,
        )
        and np.isclose(
            actual_probabilities.sum(),
            1.0,
            atol=2e-9,
            rtol=0.0,
        )
        and np.isclose(
            logsumexp(logs),
            0.0,
            atol=2e-9,
            rtol=0.0,
        ),
        "Qiskit prostorna raspodela nije dosledna ili normalizovana.",
    )

    reduced = partial_trace(
        state, list(range(POSITION_QUBITS))
    ).data
    purity = float(np.trace(reduced @ reduced).real)

    require(
        0.5 - 2e-9 <= purity <= 1.0 + 2e-9,
        "Neispravna spinorska cistoca.",
    )

    return {
        "logs": logs,
        "spin_zero_mass": float(np.vdot(upper, upper).real),
        "spin_one_mass": float(np.vdot(lower, lower).real),
        "spin_purity": purity,
    }


def self_checks():
    require(
        np.allclose(SIGMA_X @ SIGMA_X, IDENTITY_SPIN)
        and np.allclose(SIGMA_Z @ SIGMA_Z, IDENTITY_SPIN)
        and np.allclose(
            SIGMA_X @ SIGMA_Z + SIGMA_Z @ SIGMA_X,
            0.0,
            atol=1e-14,
        )
        and np.allclose(
            MOMENTUM, MOMENTUM.conj().T,
            atol=2e-12,
        ),
        "Diracova algebra ili operator impulsa nisu ispravni.",
    )

    coefficients = np.array([1.3, 0.7, 0.0, 0.0, 0.0, 0.0])
    operator = make_operator(coefficients, with_adjoint=True)
    hamiltonian = operator["hamiltonian"]

    expected_square = (
        coefficients[0] ** 2
        * np.kron(IDENTITY_SPIN, MOMENTUM @ MOMENTUM)
        + coefficients[1] ** 2 * np.eye(ACTIVE_DIM)
    )

    require(
        np.allclose(
            hamiltonian @ hamiltonian,
            expected_square,
            atol=2e-10,
            rtol=2e-10,
        )
        and np.allclose(
            operator["unitary"],
            expm(-1j * hamiltonian),
            atol=2e-10,
            rtol=2e-10,
        ),
        "Provera slobodnog Diracovog operatora nije prosla.",
    )

    fixture = np.array([
        [0, 4, 9, 13, 20, 30, 38],
        [1, 6, 10, 14, 22, 31, 37],
        [0, 5, 12, 16, 24, 33, 38],
        [2, 7, 11, 18, 23, 29, 36],
        [1, 8, 14, 19, 26, 32, 37],
        [3, 9, 15, 21, 27, 34, 38],
    ], dtype=np.int64)

    times = np.arange(1, len(fixture))
    previous = fixture[:-1]
    targets = fixture[1:]

    for window in WINDOWS:
        raw = raw_history_features(fixture, times, window)
        transform = fit_transform(raw, 3)
        features = transform_features(raw, transform)
        feature_count = features.shape[1]

        theta = initial_parameters(feature_count, 0)
        count = parameter_count(feature_count)
        models = theta.reshape(K, count)

        for slot in (0, K - 1):
            verify_qiskit(
                models[slot],
                features[0],
                int(previous[0, slot]),
            )

        value, analytic = objective(
            theta, features, previous, targets, 0.003
        )
        require(np.isfinite(value), "Nekonacan proverni gubitak.")

        selected = set(
            np.linspace(0, len(theta) - 1, 18, dtype=int).tolist()
        )

        for slot in (0, K - 1):
            selected.add(slot * count)
            selected.add(slot * count + feature_count)

            for local in range(2 * feature_count, count):
                selected.add(slot * count + local)

        step = 1e-6

        for index in sorted(selected):
            upper = theta.copy()
            lower = theta.copy()
            upper[index] += step
            lower[index] -= step

            numeric = (
                objective(
                    upper, features, previous, targets,
                    0.003, with_gradient=False,
                )[0]
                - objective(
                    lower, features, previous, targets,
                    0.003, with_gradient=False,
                )[0]
            ) / (2.0 * step)

            require(
                np.isclose(
                    analytic[index],
                    numeric,
                    atol=5e-5,
                    rtol=5e-4,
                ),
                "Koordinatna provera Diracovog gradijenta nije prosla.",
            )

        for direction_index in range(3):
            direction = np.sin(
                (np.arange(len(theta)) + 1.0)
                * (0.31 + 0.17 * direction_index)
            )
            direction /= np.linalg.norm(direction)

            numeric = (
                objective(
                    theta + step * direction,
                    features, previous, targets,
                    0.003, with_gradient=False,
                )[0]
                - objective(
                    theta - step * direction,
                    features, previous, targets,
                    0.003, with_gradient=False,
                )[0]
            ) / (2.0 * step)

            require(
                np.isclose(
                    analytic @ direction,
                    numeric,
                    atol=5e-5,
                    rtol=5e-4,
                ),
                "Usmerena provera Diracovog gradijenta nije prosla.",
            )

        changed = fixture.copy()
        changed[4:] = changed[4:][::-1]

        changed_raw = raw_history_features(
            changed, times, window
        )
        changed_transform = fit_transform(changed_raw, 3)

        require(
            np.array_equal(raw[:4], changed_raw[:4])
            and np.array_equal(
                transform["keep"], changed_transform["keep"]
            )
            and np.allclose(
                transform["mean"], changed_transform["mean"]
            )
            and np.allclose(
                transform["scale"], changed_transform["scale"]
            ),
            "Buduci podaci uticu na ranije karakteristike.",
        )

    toy = np.exp(
        0.5 * np.sin(np.arange(15, dtype=float).reshape(3, 5))
    )
    toy /= toy.sum(axis=1, keepdims=True)
    toy_logs = np.log(toy)
    joint = joint_distribution(toy_logs)

    combinations = list(itertools.combinations(range(5), 3))
    scores = np.array([
        toy_logs[np.arange(3), combination].sum()
        for combination in combinations
    ])
    brute_marginals = np.zeros_like(toy)

    for combination, mass in zip(
        combinations, np.exp(scores - logsumexp(scores))
    ):
        brute_marginals[np.arange(3), combination] += mass

    require(
        np.isclose(
            joint["log_z"], logsumexp(scores),
            atol=1e-12,
        )
        and np.isclose(
            toy_logs[np.arange(3), joint["path"]].sum(),
            scores.max(),
            atol=1e-12,
        )
        and np.allclose(
            joint["marginals"],
            brute_marginals,
            atol=1e-12,
            rtol=0.0,
        ),
        "Potpuna DP provera nije prosla.",
    )

    sole = np.full((3, 5), -np.inf)
    sole[np.arange(3), [0, 2, 4]] = 0.0
    sole_joint = joint_distribution(sole)

    require(
        np.array_equal(sole_joint["path"], [0, 2, 4])
        and np.isclose(sole_joint["log_z"], 0.0),
        "Provera jedine legalne kombinacije nije prosla.",
    )

    impossible = np.full((3, 5), -np.inf)
    impossible[:, 0] = 0.0

    try:
        joint_distribution(impossible)
    except DistributionError:
        pass
    else:
        raise RuntimeError("Nulta legalna masa nije prijavljena.")

    require(
        np.isfinite(log_amplitudes(np.array([1e-180 + 0j]))[0]),
        "Izgubljena mala amplituda.",
    )

    print("Ugradjene Dirac, Qiskit, gradijent, hronologija i DP provere su prosle.")


def longest_consecutive(values):
    longest = 1
    current = 1

    for difference in np.diff(values):
        current = current + 1 if difference == 1 else 1
        longest = max(longest, current)

    return longest


def process_dataset(dataset):
    data = dataset["data"]
    path = dataset["path"]

    times = np.arange(1, len(data))
    previous = data[:-1]
    targets = data[1:]
    total = len(targets)

    train_end = int(TRAIN_FRACTION * total)
    validation_end = int(VALIDATION_END_FRACTION * total)

    require(
        0 < train_end < validation_end < total,
        "Nedovoljno podataka za hronolosku podelu.",
    )

    print()
    print("=" * 72)
    print(path.name)
    print(f"Stvarni broj kola: {len(data)}")
    print(f"SHA256: {dataset['digest']}")
    print(f"Trening prelaza: {train_end}")
    print(f"Validacionih prelaza: {validation_end - train_end}")
    print(f"Test prelaza: {total - validation_end}")

    best = None

    for window in WINDOWS:
        raw = raw_history_features(data, times, window)
        transform = fit_transform(raw, train_end)
        features = transform_features(raw, transform)

        for weight in REGULARIZATIONS:
            label = f"VALIDACIJA prozor={window}, L2={weight}"

            models = fit_models(
                features[:train_end],
                previous[:train_end],
                targets[:train_end],
                weight,
                label,
            )

            try:
                metrics = evaluate(
                    models,
                    features[train_end:validation_end],
                    previous[train_end:validation_end],
                    targets[train_end:validation_end],
                )
            except DistributionError as exc:
                print(f"Kandidat odbijen: {exc}")
                continue

            print_metrics(label, metrics)

            if np.isfinite(metrics["nll"]) and (
                best is None or metrics["nll"] < best["nll"]
            ):
                best = {
                    "window": window,
                    "weight": weight,
                    "nll": metrics["nll"],
                }

    require(best is not None, "Nema validnog Diracovog kandidata.")

    window = best["window"]
    weight = best["weight"]
    raw = raw_history_features(data, times, window)

    print()
    print(f"Izabrani prozor: {window}")
    print(f"Izabrana regularizacija: {weight}")

    test_transform = fit_transform(raw, validation_end)
    test_features = transform_features(raw, test_transform)

    test_models = fit_models(
        test_features[:validation_end],
        previous[:validation_end],
        targets[:validation_end],
        weight,
        "UCENJE ZA ODVOJENI TEST",
    )
    metrics = evaluate(
        test_models,
        test_features[validation_end:],
        previous[validation_end:],
        targets[validation_end:],
    )
    print_metrics("ODVOJENI HRONOLOSKI TEST", metrics)

    final_transform = fit_transform(raw, len(raw))
    final_features = transform_features(raw, final_transform)

    final_models = fit_models(
        final_features,
        previous,
        targets,
        weight,
        "ZAVRSNO UCENJE NA CELOJ ISTORIJI",
    )

    next_raw = raw_history_features(
        data, np.array([len(data)]), window
    )
    next_features = transform_features(
        next_raw, final_transform
    )
    next_previous = data[-1:].copy()

    feature_count = next_features.shape[1]
    next_logs = np.empty((K, N))
    operators = []

    print()
    print("NAUCENI DIRACOVI PARAMETRI I SPINOR")

    for slot in range(K):
        coefficients = final_models[slot, 2 * feature_count:]
        operators.append(make_operator(coefficients))

        verification = verify_qiskit(
            final_models[slot],
            next_features[0],
            int(next_previous[0, slot]),
        )
        next_logs[slot] = verification["logs"]

        print(
            f"Registar {slot + 1}, "
            f"c={coefficients[0]:.6f}, "
            f"mu={coefficients[1]:.6f}"
        )
        print(
            f"Spinor 0 masa={verification['spin_zero_mass']:.8f}, "
            f"spinor 1 masa={verification['spin_one_mass']:.8f}, "
            f"cistoca={verification['spin_purity']:.8f}"
        )

    numerical_logs = predict_logs(
        final_models,
        next_features,
        next_previous,
        operators=operators,
    )[0]

    require(
        np.allclose(
            np.exp(next_logs),
            np.exp(numerical_logs),
            atol=2e-9,
            rtol=2e-8,
        ),
        "Zavrsne Qiskit raspodele nisu dosledne.",
    )

    distribution = joint_distribution(next_logs)
    combination = distribution["path"] + 1

    require(
        len(set(combination.tolist())) == K
        and np.all((combination >= 1) & (combination <= N)),
        "Nelegalna NEXT kombinacija.",
    )

    probe_times = np.unique(
        np.array([1, max(1, len(data) // 2), len(data)])
    )
    probe_features = transform_features(
        raw_history_features(data, probe_times, window),
        final_transform,
    )
    probe_previous = data[probe_times - 1]

    probe_logs = predict_logs(
        final_models,
        probe_features,
        probe_previous,
        operators=operators,
    )
    variation = float(np.max(np.abs(
        np.exp(probe_logs) - np.exp(next_logs)[None, :, :]
    )))

    centered_features = np.zeros_like(next_features)
    centered_features[:, 0] = 1.0

    centered_logs = predict_logs(
        final_models,
        centered_features,
        next_previous,
        operators=operators,
    )[0]
    centered_distribution = joint_distribution(centered_logs)

    spinor_history_effect = float(np.max(np.abs(
        distribution["marginals"]
        - centered_distribution["marginals"]
    )))

    print()
    print(f"Log legalne mase: {distribution['log_z']:.9f}")
    print(
        f"Log uslovne mase NEXT: "
        f"{distribution['log_mode_probability']:.9f}"
    )
    print(f"Promena kroz stvarne istorijske kontekste: {variation:.9e}")
    print(
        f"Uticaj istorijskih spinorskih ulaza na marginale: "
        f"{spinor_history_effect:.9e}"
    )

    if spinor_history_effect < 1e-8:
        print("Napomena: veoma slab dodatni uticaj spinorskih karakteristika.")

    if np.array_equal(combination, data[-1] + 1):
        print("Upozorenje: NEXT ponavlja poslednje kolo.")

    run = longest_consecutive(combination)

    if run >= 5:
        print(f"Upozorenje: niz od {run} uzastopnih brojeva.")

    span = int(combination[-1] - combination[0])
    historical_spans = data[:, -1] - data[:, 0]
    percentile = 100.0 * float(np.mean(historical_spans <= span))

    print(f"Raspon NEXT kombinacije: {span}")
    print(f"Udeo istorijskih raspona <= NEXT raspon: {percentile:.3f}%")

    if percentile <= 1.0:
        print("Upozorenje: izrazito grupisana NEXT kombinacija.")

    return {
        "name": path.name,
        "digest": dataset["digest"],
        "next": combination,
        "probabilities": np.exp(next_logs),
        "marginals": distribution["marginals"],
    }


def main():
    print("Quantum Dirac V1")
    print(f"Qiskit: {qiskit.__version__}")
    print("Sedam zasebnih registara sa po sest prostornih i jednim spinorskim kubitom.")
    print("Diracov kineticki clan, maseni clan i nauceni potencijal.")
    print("Spektralna periodicna prostorna mreza: 39 stanja.")
    print("Preostalih 25 prostornih stanja je neaktivno.")
    print("Obe spinorske komponente ulaze u Bornovu raspodelu.")
    print("Zajednicki uslovni NLL svih legalnih kombinacija.")
    print("Analiticki Frechetov adjoint gradijent.")
    print("L-BFGS-B je klasicni optimizator kvantnih parametara.")
    print(f"Interval c: {SPEED_BOUNDS}")
    print(f"Interval mu=m*c^2: {MASS_ENERGY_BOUNDS}")
    print(f"Interval koeficijenata potencijala: {POTENTIAL_BOUNDS}")
    print()

    self_checks()

    results = [
        process_dataset(load_csv(path))
        for path in FILES
    ]

    print()
    print("=" * 72)
    print("POREDJENJE NEZAVISNIH REZULTATA")

    for first, second in itertools.combinations(results, 2):
        if first["digest"] == second["digest"]:
            print(
                f"Upozorenje: isti sadrzaj fajlova "
                f"{first['name']} i {second['name']}."
            )

        if np.array_equal(first["next"], second["next"]):
            print()
            print(
                f"Upozorenje: isti NEXT za "
                f"{first['name']} i {second['name']}."
            )
            print(
                "Najveca razlika registarskih raspodela: "
                f"{np.max(np.abs(first['probabilities'] - second['probabilities'])):.9e}"
            )
            print(
                "Najveca razlika legalnih marginala: "
                f"{np.max(np.abs(first['marginals'] - second['marginals'])):.9e}"
            )

    print()
    print("=" * 72)

    for result in results:
        numbers = " ".join(
            str(int(value)) for value in result["next"]
        )
        print(f"{result['name']} NEXT: {numbers}")


if __name__ == "__main__":
    main()


    
"""
Quantum Dirac V1
Qiskit: 1.4.4
Sedam zasebnih registara sa po sest prostornih i jednim spinorskim kubitom.
Diracov kineticki clan, maseni clan i nauceni potencijal.
Spektralna periodicna prostorna mreza: 39 stanja.
Preostalih 25 prostornih stanja je neaktivno.
Obe spinorske komponente ulaze u Bornovu raspodelu.
Zajednicki uslovni NLL svih legalnih kombinacija.
Analiticki Frechetov adjoint gradijent.
L-BFGS-B je klasicni optimizator kvantnih parametara.
Interval c: (0.25, 20.0)
Interval mu=m*c^2: (0.05, 15.0)
Interval koeficijenata potencijala: (-8.0, 8.0)

Ugradjene Dirac, Qiskit, gradijent, hronologija i DP provere su prosle.

========================================================================
loto7_4716_k89.csv
Stvarni broj kola: 4716
SHA256: deb02ea741f500e3270402730ed89565f1c9b4618d53e5a08fea2b7c2c2af31a
Trening prelaza: 3300
Validacionih prelaza: 707
Test prelaza: 708
VALIDACIJA prozor=1, L2=0.001
Trening prelaza: 3300
Broj karakteristika sa interceptom: 8
L2: 0.001
Deterministicki pocetak 1/2
Procena 10, najbolji NLL+L2=23.534433419
Procena 20, najbolji NLL+L2=20.181953264
Procena 30, najbolji NLL+L2=18.980487536
Procena 40, najbolji NLL+L2=18.614688551
Procena 50, najbolji NLL+L2=18.382093648
Procena 60, najbolji NLL+L2=18.275703653
Procena 70, najbolji NLL+L2=18.229340961
Procena 80, najbolji NLL+L2=18.204537358
Procena 90, najbolji NLL+L2=18.188147252
Procena 100, najbolji NLL+L2=18.163545671
Procena 110, najbolji NLL+L2=18.140751671
Procena 120, najbolji NLL+L2=18.127257419
Procena 130, najbolji NLL+L2=18.120500777
Procena 140, najbolji NLL+L2=18.113352086
Procena 150, najbolji NLL+L2=18.106718219
Procena 160, najbolji NLL+L2=18.104885222
Procena 170, najbolji NLL+L2=18.103440143
Procena 180, najbolji NLL+L2=18.101882250
Procena 190, najbolji NLL+L2=18.101104277
Procena 200, najbolji NLL+L2=18.101013004
Procena 210, najbolji NLL+L2=18.100981198
Sacuvani NLL+L2: 18.100977425
Max|projektovani gradijent|: 3.392e-04
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Napomena: mali projektovani gradijent nije potvrdjen.
Deterministicki pocetak 2/2
Procena 10, najbolji NLL+L2=20.287526537
Procena 20, najbolji NLL+L2=18.388133327
Procena 30, najbolji NLL+L2=17.963511046
Procena 40, najbolji NLL+L2=17.745615043
Procena 50, najbolji NLL+L2=17.647480834
Procena 60, najbolji NLL+L2=17.584176852
Procena 70, najbolji NLL+L2=17.497923006
Procena 80, najbolji NLL+L2=17.452901814
Procena 90, najbolji NLL+L2=17.406265489
Procena 100, najbolji NLL+L2=17.362943194
Procena 110, najbolji NLL+L2=17.348273826
Procena 120, najbolji NLL+L2=17.332324090
Procena 130, najbolji NLL+L2=17.315606068
Procena 140, najbolji NLL+L2=17.309501187
Procena 150, najbolji NLL+L2=17.304780237
Procena 160, najbolji NLL+L2=17.301559130
Procena 170, najbolji NLL+L2=17.298507795
Procena 180, najbolji NLL+L2=17.296627364
Procena 190, najbolji NLL+L2=17.294958991
Procena 200, najbolji NLL+L2=17.293439228
Procena 210, najbolji NLL+L2=17.287101264
Sacuvani NLL+L2: 17.287101264
Max|projektovani gradijent|: 1.493e-02
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Napomena: mali projektovani gradijent nije potvrdjen.
VALIDACIJA prozor=1, L2=0.001
Srednji uslovni NLL: 17.378014123
Srednji broj pogodaka: 1.168317
Ciljevi sa nultom masom: 0
Broj kola po pogocima 0..7: [182, 292, 173, 52, 8, 0, 0, 0]
VALIDACIJA prozor=1, L2=0.01
Trening prelaza: 3300
Broj karakteristika sa interceptom: 8
L2: 0.01
Deterministicki pocetak 1/2
Procena 10, najbolji NLL+L2=23.535586800
Procena 20, najbolji NLL+L2=20.223269153
Procena 30, najbolji NLL+L2=19.133259368
Procena 40, najbolji NLL+L2=18.662503277
Procena 50, najbolji NLL+L2=18.497429705
Procena 60, najbolji NLL+L2=18.406575615
Procena 70, najbolji NLL+L2=18.361496409
Procena 80, najbolji NLL+L2=18.295136038
Procena 90, najbolji NLL+L2=18.268870514
Procena 100, najbolji NLL+L2=18.255622331
Procena 110, najbolji NLL+L2=18.247314227
Procena 120, najbolji NLL+L2=18.235502911
Procena 130, najbolji NLL+L2=18.212535075
Procena 140, najbolji NLL+L2=18.204436543
Procena 150, najbolji NLL+L2=18.202791788
Procena 160, najbolji NLL+L2=18.201993868
Procena 170, najbolji NLL+L2=18.201748715
Procena 180, najbolji NLL+L2=18.201194983
Procena 190, najbolji NLL+L2=18.196038666
Procena 200, najbolji NLL+L2=18.194691110
Procena 210, najbolji NLL+L2=18.192659943
Sacuvani NLL+L2: 18.189362234
Max|projektovani gradijent|: 1.354e-02
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Napomena: mali projektovani gradijent nije potvrdjen.
Deterministicki pocetak 2/2
Procena 10, najbolji NLL+L2=20.347087721
Procena 20, najbolji NLL+L2=18.333727916
Procena 30, najbolji NLL+L2=17.844975701
Procena 40, najbolji NLL+L2=17.694138551
Procena 50, najbolji NLL+L2=17.604449359
Procena 60, najbolji NLL+L2=17.550387579
Procena 70, najbolji NLL+L2=17.483141475
Procena 80, najbolji NLL+L2=17.413651661
Procena 90, najbolji NLL+L2=17.392894687
Procena 100, najbolji NLL+L2=17.354187183
Procena 110, najbolji NLL+L2=17.342644775
Procena 120, najbolji NLL+L2=17.339002702
Procena 130, najbolji NLL+L2=17.334860186
Procena 140, najbolji NLL+L2=17.326873641
Procena 150, najbolji NLL+L2=17.317647322
Procena 160, najbolji NLL+L2=17.291862505
Procena 170, najbolji NLL+L2=17.279803003
Procena 180, najbolji NLL+L2=17.275809977
Procena 190, najbolji NLL+L2=17.272428618
Procena 200, najbolji NLL+L2=17.270028293
Sacuvani NLL+L2: 17.269135529
Max|projektovani gradijent|: 4.339e-03
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Napomena: mali projektovani gradijent nije potvrdjen.
VALIDACIJA prozor=1, L2=0.01
Srednji uslovni NLL: 17.359061064
Srednji broj pogodaka: 1.214993
Ciljevi sa nultom masom: 0
Broj kola po pogocima 0..7: [165, 301, 179, 50, 10, 2, 0, 0]
VALIDACIJA prozor=3, L2=0.001
Trening prelaza: 3300
Broj karakteristika sa interceptom: 24
L2: 0.001
Deterministicki pocetak 1/2
Procena 10, najbolji NLL+L2=23.392489667
Procena 20, najbolji NLL+L2=20.230053785
Procena 30, najbolji NLL+L2=18.968444368
Procena 40, najbolji NLL+L2=18.560455456
Procena 50, najbolji NLL+L2=18.396934305
Procena 60, najbolji NLL+L2=18.323655015
Procena 70, najbolji NLL+L2=18.300834191
Procena 80, najbolji NLL+L2=18.271746404
Procena 90, najbolji NLL+L2=18.250309386
Procena 100, najbolji NLL+L2=18.220816311
Procena 110, najbolji NLL+L2=18.207896801
Procena 120, najbolji NLL+L2=18.199377249
Procena 130, najbolji NLL+L2=18.192334061
Procena 140, najbolji NLL+L2=18.183926325
Procena 150, najbolji NLL+L2=18.176852832
Procena 160, najbolji NLL+L2=18.172230114
Procena 170, najbolji NLL+L2=18.169552135
Procena 180, najbolji NLL+L2=18.167776294
Procena 190, najbolji NLL+L2=18.166705321
Procena 200, najbolji NLL+L2=18.165610901
Sacuvani NLL+L2: 18.164535265
Max|projektovani gradijent|: 1.983e-02
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Napomena: mali projektovani gradijent nije potvrdjen.
Deterministicki pocetak 2/2
Procena 10, najbolji NLL+L2=21.176802951
Procena 20, najbolji NLL+L2=19.150553016
Procena 30, najbolji NLL+L2=18.070219964
Procena 40, najbolji NLL+L2=17.833801835
Procena 50, najbolji NLL+L2=17.743825209
Procena 60, najbolji NLL+L2=17.668558170
Procena 70, najbolji NLL+L2=17.607368875
Procena 80, najbolji NLL+L2=17.568712015
Procena 90, najbolji NLL+L2=17.553245216
Procena 100, najbolji NLL+L2=17.541484913
Procena 110, najbolji NLL+L2=17.531211172
Procena 120, najbolji NLL+L2=17.520031572
Procena 130, najbolji NLL+L2=17.509394110
Procena 140, najbolji NLL+L2=17.500997108
Procena 150, najbolji NLL+L2=17.495294452
Procena 160, najbolji NLL+L2=17.492429487
Procena 170, najbolji NLL+L2=17.490029998
Procena 180, najbolji NLL+L2=17.486345039
Procena 190, najbolji NLL+L2=17.477690556
Procena 200, najbolji NLL+L2=17.469995656
Sacuvani NLL+L2: 17.469059277
Max|projektovani gradijent|: 1.219e-02
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Napomena: mali projektovani gradijent nije potvrdjen.
VALIDACIJA prozor=3, L2=0.001
Srednji uslovni NLL: 17.834513451
Srednji broj pogodaka: 1.230552
Ciljevi sa nultom masom: 0
Broj kola po pogocima 0..7: [159, 289, 205, 46, 7, 1, 0, 0]
VALIDACIJA prozor=3, L2=0.01
Trening prelaza: 3300
Broj karakteristika sa interceptom: 24
L2: 0.01
Deterministicki pocetak 1/2
Procena 10, najbolji NLL+L2=23.393033654
Procena 20, najbolji NLL+L2=20.272448389
Procena 30, najbolji NLL+L2=19.069750052
Procena 40, najbolji NLL+L2=18.762245967
Procena 50, najbolji NLL+L2=18.610438840
Procena 60, najbolji NLL+L2=18.531872007
Procena 70, najbolji NLL+L2=18.497540744
Procena 80, najbolji NLL+L2=18.461788281
Procena 90, najbolji NLL+L2=18.430815413
Procena 100, najbolji NLL+L2=18.406321625
Procena 110, najbolji NLL+L2=18.391501415
Procena 120, najbolji NLL+L2=18.373862455
Procena 130, najbolji NLL+L2=18.362632168
Procena 140, najbolji NLL+L2=18.353782010
Procena 150, najbolji NLL+L2=18.348509486
Procena 160, najbolji NLL+L2=18.341943162
Procena 170, najbolji NLL+L2=18.336879950
Procena 180, najbolji NLL+L2=18.330762122
Procena 190, najbolji NLL+L2=18.325486828
Procena 200, najbolji NLL+L2=18.323071855
Sacuvani NLL+L2: 18.319455597
Max|projektovani gradijent|: 1.303e-02
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Napomena: mali projektovani gradijent nije potvrdjen.
Deterministicki pocetak 2/2
Procena 10, najbolji NLL+L2=21.176790324
Procena 20, najbolji NLL+L2=19.145443648
Procena 30, najbolji NLL+L2=18.072994481
Procena 40, najbolji NLL+L2=17.830664938
Procena 50, najbolji NLL+L2=17.748208840
Procena 60, najbolji NLL+L2=17.679788315
Procena 70, najbolji NLL+L2=17.629021337
Procena 80, najbolji NLL+L2=17.587871663
Procena 90, najbolji NLL+L2=17.569275014
Procena 100, najbolji NLL+L2=17.556284193
Procena 110, najbolji NLL+L2=17.540806373
Procena 120, najbolji NLL+L2=17.524507770
Procena 130, najbolji NLL+L2=17.513618738
Procena 140, najbolji NLL+L2=17.509389962
Procena 150, najbolji NLL+L2=17.503476901
Procena 160, najbolji NLL+L2=17.496353789
Procena 170, najbolji NLL+L2=17.491476132
Procena 180, najbolji NLL+L2=17.488434014
Procena 190, najbolji NLL+L2=17.482687615
Procena 200, najbolji NLL+L2=17.477185388
Sacuvani NLL+L2: 17.476561534
Max|projektovani gradijent|: 6.364e-03
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Napomena: mali projektovani gradijent nije potvrdjen.
VALIDACIJA prozor=3, L2=0.01
Srednji uslovni NLL: 17.838151058
Srednji broj pogodaka: 1.265912
Ciljevi sa nultom masom: 0
Broj kola po pogocima 0..7: [157, 282, 200, 60, 7, 1, 0, 0]

Izabrani prozor: 1
Izabrana regularizacija: 0.01
UCENJE ZA ODVOJENI TEST
Trening prelaza: 4007
Broj karakteristika sa interceptom: 8
L2: 0.01
Deterministicki pocetak 1/2
Procena 10, najbolji NLL+L2=23.601545626
Procena 20, najbolji NLL+L2=21.063172504
Procena 30, najbolji NLL+L2=19.468949549
Procena 40, najbolji NLL+L2=18.968223702
Procena 50, najbolji NLL+L2=18.738386212
Procena 60, najbolji NLL+L2=18.675904075
Procena 70, najbolji NLL+L2=18.646368530
Procena 80, najbolji NLL+L2=18.629161439
Procena 90, najbolji NLL+L2=18.619979336
Procena 100, najbolji NLL+L2=18.614395324
Procena 110, najbolji NLL+L2=18.610884516
Procena 120, najbolji NLL+L2=18.597503769
Procena 130, najbolji NLL+L2=18.574065969
Procena 140, najbolji NLL+L2=18.569683071
Procena 150, najbolji NLL+L2=18.567609500
Procena 160, najbolji NLL+L2=18.566290402
Procena 170, najbolji NLL+L2=18.565895708
Procena 180, najbolji NLL+L2=18.565553329
Procena 190, najbolji NLL+L2=18.565068733
Procena 200, najbolji NLL+L2=18.564484764
Procena 210, najbolji NLL+L2=18.564302957
Sacuvani NLL+L2: 18.564264099
Max|projektovani gradijent|: 1.088e-03
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Napomena: mali projektovani gradijent nije potvrdjen.
Deterministicki pocetak 2/2
Procena 10, najbolji NLL+L2=21.553355275
Procena 20, najbolji NLL+L2=19.340194835
Procena 30, najbolji NLL+L2=18.268873301
Procena 40, najbolji NLL+L2=17.955727777
Procena 50, najbolji NLL+L2=17.809175578
Procena 60, najbolji NLL+L2=17.749203082
Procena 70, najbolji NLL+L2=17.730071668
Procena 80, najbolji NLL+L2=17.707846382
Procena 90, najbolji NLL+L2=17.690293648
Procena 100, najbolji NLL+L2=17.678559077
Procena 110, najbolji NLL+L2=17.669606093
Procena 120, najbolji NLL+L2=17.662862298
Procena 130, najbolji NLL+L2=17.656876591
Procena 140, najbolji NLL+L2=17.648094254
Procena 150, najbolji NLL+L2=17.642358860
Procena 160, najbolji NLL+L2=17.641544779
Procena 170, najbolji NLL+L2=17.641365677
Procena 180, najbolji NLL+L2=17.641244079
Procena 190, najbolji NLL+L2=17.641124085
Procena 200, najbolji NLL+L2=17.641032448
Sacuvani NLL+L2: 17.640991218
Max|projektovani gradijent|: 2.102e-03
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Napomena: mali projektovani gradijent nije potvrdjen.
ODVOJENI HRONOLOSKI TEST
Srednji uslovni NLL: 17.866535879
Srednji broj pogodaka: 1.290960
Ciljevi sa nultom masom: 0
Broj kola po pogocima 0..7: [150, 280, 211, 56, 11, 0, 0, 0]
ZAVRSNO UCENJE NA CELOJ ISTORIJI
Trening prelaza: 4715
Broj karakteristika sa interceptom: 8
L2: 0.01
Deterministicki pocetak 1/2
Procena 10, najbolji NLL+L2=23.621432068
Procena 20, najbolji NLL+L2=20.161362142
Procena 30, najbolji NLL+L2=18.826366882
Procena 40, najbolji NLL+L2=18.514822993
Procena 50, najbolji NLL+L2=18.357375689
Procena 60, najbolji NLL+L2=18.280557888
Procena 70, najbolji NLL+L2=18.232088401
Procena 80, najbolji NLL+L2=18.217925979
Procena 90, najbolji NLL+L2=18.209105847
Procena 100, najbolji NLL+L2=18.197339489
Procena 110, najbolji NLL+L2=18.183555845
Procena 120, najbolji NLL+L2=18.179476859
Procena 130, najbolji NLL+L2=18.175155056
Procena 140, najbolji NLL+L2=18.167490623
Procena 150, najbolji NLL+L2=18.162265993
Procena 160, najbolji NLL+L2=18.159356075
Procena 170, najbolji NLL+L2=18.158657585
Procena 180, najbolji NLL+L2=18.157631789
Procena 190, najbolji NLL+L2=18.154214148
Procena 200, najbolji NLL+L2=18.151836812
Sacuvani NLL+L2: 18.150120702
Max|projektovani gradijent|: 5.348e-03
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Napomena: mali projektovani gradijent nije potvrdjen.
Deterministicki pocetak 2/2
Procena 10, najbolji NLL+L2=20.110303666
Procena 20, najbolji NLL+L2=19.550104172
Procena 30, najbolji NLL+L2=19.177108305
Procena 40, najbolji NLL+L2=18.891114941
Procena 50, najbolji NLL+L2=18.372043470
Procena 60, najbolji NLL+L2=18.170766990
Procena 70, najbolji NLL+L2=18.027647030
Procena 80, najbolji NLL+L2=17.899253772
Procena 90, najbolji NLL+L2=17.812514515
Procena 100, najbolji NLL+L2=17.778185235
Procena 110, najbolji NLL+L2=17.777913293
Procena 120, najbolji NLL+L2=17.749362011
Procena 130, najbolji NLL+L2=17.741336453
Procena 140, najbolji NLL+L2=17.738698680
Procena 150, najbolji NLL+L2=17.737130855
Procena 160, najbolji NLL+L2=17.735343812
Procena 170, najbolji NLL+L2=17.734030512
Procena 180, najbolji NLL+L2=17.733055458
Procena 190, najbolji NLL+L2=17.732230699
Procena 200, najbolji NLL+L2=17.729757730
Procena 210, najbolji NLL+L2=17.718802699
Procena 220, najbolji NLL+L2=17.711510585
Sacuvani NLL+L2: 17.710459339
Max|projektovani gradijent|: 7.458e-03
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Napomena: mali projektovani gradijent nije potvrdjen.

NAUCENI DIRACOVI PARAMETRI I SPINOR
Registar 1, c=14.407335, mu=7.846888
Spinor 0 masa=0.37891732, spinor 1 masa=0.62108268, cistoca=0.57865001
Registar 2, c=13.991764, mu=12.516809
Spinor 0 masa=0.60882363, spinor 1 masa=0.39117637, cistoca=0.60300749
Registar 3, c=7.940843, mu=6.526615
Spinor 0 masa=0.46268211, spinor 1 masa=0.53731789, cistoca=0.53196699
Registar 4, c=12.254341, mu=3.510239
Spinor 0 masa=0.55210223, spinor 1 masa=0.44789777, cistoca=0.61375205
Registar 5, c=8.673963, mu=4.623807
Spinor 0 masa=0.38774578, spinor 1 masa=0.61225422, cistoca=0.54720207
Registar 6, c=7.701513, mu=4.108228
Spinor 0 masa=0.59109894, spinor 1 masa=0.40890106, cistoca=0.54517890
Registar 7, c=15.963231, mu=7.430851
Spinor 0 masa=0.41539956, spinor 1 masa=0.58460044, cistoca=0.52627741

Log legalne mase: -11.001491047
Log uslovne mase NEXT: -10.523688323
Promena kroz stvarne istorijske kontekste: 3.803011635e-01
Uticaj istorijskih spinorskih ulaza na marginale: 1.258453987e-01
Upozorenje: niz od 5 uzastopnih brojeva.
Raspon NEXT kombinacije: 19
Udeo istorijskih raspona <= NEXT raspon: 3.753%

========================================================================
loto7_4716_k89_loto_2980.csv
Stvarni broj kola: 2979
SHA256: f1cd81cd04660d50d9058675924da34346fc006ce8dabd84d52266c66afe17d1
Trening prelaza: 2084
Validacionih prelaza: 447
Test prelaza: 447
VALIDACIJA prozor=1, L2=0.001
Trening prelaza: 2084
Broj karakteristika sa interceptom: 8
L2: 0.001
Deterministicki pocetak 1/2
Procena 10, najbolji NLL+L2=23.540292742
Procena 20, najbolji NLL+L2=20.897449756
Procena 30, najbolji NLL+L2=18.794721879
Procena 40, najbolji NLL+L2=18.464462211
Procena 50, najbolji NLL+L2=18.271730327
Procena 60, najbolji NLL+L2=18.171917394
Procena 70, najbolji NLL+L2=18.142261287
Procena 80, najbolji NLL+L2=18.115955061
Procena 90, najbolji NLL+L2=18.101460001
Procena 100, najbolji NLL+L2=18.094079439
Procena 110, najbolji NLL+L2=18.089853740
Procena 120, najbolji NLL+L2=18.086183502
Procena 130, najbolji NLL+L2=18.079534586
Procena 140, najbolji NLL+L2=18.075791078
Procena 150, najbolji NLL+L2=18.073405258
Procena 160, najbolji NLL+L2=18.071450573
Procena 170, najbolji NLL+L2=18.071138996
Procena 180, najbolji NLL+L2=18.071066260
Procena 190, najbolji NLL+L2=18.071041808
Procena 200, najbolji NLL+L2=18.071029074
Sacuvani NLL+L2: 18.071023845
Max|projektovani gradijent|: 1.939e-04
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Napomena: mali projektovani gradijent nije potvrdjen.
Deterministicki pocetak 2/2
Procena 10, najbolji NLL+L2=19.774904344
Procena 20, najbolji NLL+L2=19.357864450
Procena 30, najbolji NLL+L2=18.855085167
Procena 40, najbolji NLL+L2=18.639995154
Procena 50, najbolji NLL+L2=18.501738322
Procena 60, najbolji NLL+L2=18.320987598
Procena 70, najbolji NLL+L2=18.226007260
Procena 80, najbolji NLL+L2=18.123284927
Procena 90, najbolji NLL+L2=18.093547396
Procena 100, najbolji NLL+L2=18.040852346
Procena 110, najbolji NLL+L2=17.948259880
Procena 120, najbolji NLL+L2=17.875885722
Procena 130, najbolji NLL+L2=17.845740189
Procena 140, najbolji NLL+L2=17.819760013
Procena 150, najbolji NLL+L2=17.789552931
Procena 160, najbolji NLL+L2=17.770982818
Procena 170, najbolji NLL+L2=17.755326286
Procena 180, najbolji NLL+L2=17.743805014
Procena 190, najbolji NLL+L2=17.739790976
Procena 200, najbolji NLL+L2=17.737076389
Procena 210, najbolji NLL+L2=17.735701057
Sacuvani NLL+L2: 17.735347764
Max|projektovani gradijent|: 1.282e-03
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Napomena: mali projektovani gradijent nije potvrdjen.
VALIDACIJA prozor=1, L2=0.001
Srednji uslovni NLL: 18.012483944
Srednji broj pogodaka: 1.241611
Ciljevi sa nultom masom: 0
Broj kola po pogocima 0..7: [104, 175, 127, 38, 3, 0, 0, 0]
VALIDACIJA prozor=1, L2=0.01
Trening prelaza: 2084
Broj karakteristika sa interceptom: 8
L2: 0.01
Deterministicki pocetak 1/2
Procena 10, najbolji NLL+L2=23.541217146
Procena 20, najbolji NLL+L2=20.914138746
Procena 30, najbolji NLL+L2=19.256340793
Procena 40, najbolji NLL+L2=18.556623912
Procena 50, najbolji NLL+L2=18.215784728
Procena 60, najbolji NLL+L2=18.062830421
Procena 70, najbolji NLL+L2=17.998117321
Procena 80, najbolji NLL+L2=17.909001389
Procena 90, najbolji NLL+L2=17.857002466
Procena 100, najbolji NLL+L2=17.834988185
Procena 110, najbolji NLL+L2=17.806918639
Procena 120, najbolji NLL+L2=17.793772004
Procena 130, najbolji NLL+L2=17.783182099
Procena 140, najbolji NLL+L2=17.770388281
Procena 150, najbolji NLL+L2=17.766682069
Procena 160, najbolji NLL+L2=17.765045749
Procena 170, najbolji NLL+L2=17.763366187
Procena 180, najbolji NLL+L2=17.761607376
Procena 190, najbolji NLL+L2=17.760029554
Procena 200, najbolji NLL+L2=17.759552092
Procena 210, najbolji NLL+L2=17.759449143
Sacuvani NLL+L2: 17.759447611
Max|projektovani gradijent|: 2.002e-04
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Napomena: mali projektovani gradijent nije potvrdjen.
Deterministicki pocetak 2/2
Procena 10, najbolji NLL+L2=19.421320761
Procena 20, najbolji NLL+L2=19.098388671
Procena 30, najbolji NLL+L2=18.866776152
Procena 40, najbolji NLL+L2=18.511333996
Procena 50, najbolji NLL+L2=18.312024371
Procena 60, najbolji NLL+L2=18.128416490
Procena 70, najbolji NLL+L2=17.964414659
Procena 80, najbolji NLL+L2=17.899232760
Procena 90, najbolji NLL+L2=17.872051003
Procena 100, najbolji NLL+L2=17.857837312
Procena 110, najbolji NLL+L2=17.845818760
Procena 120, najbolji NLL+L2=17.829186557
Procena 130, najbolji NLL+L2=17.822015929
Procena 140, najbolji NLL+L2=17.816278562
Procena 150, najbolji NLL+L2=17.811972175
Procena 160, najbolji NLL+L2=17.810796583
Procena 170, najbolji NLL+L2=17.810271620
Procena 180, najbolji NLL+L2=17.809877940
Procena 190, najbolji NLL+L2=17.809476084
Procena 200, najbolji NLL+L2=17.808964302
Procena 210, najbolji NLL+L2=17.807877433
Sacuvani NLL+L2: 17.807050464
Max|projektovani gradijent|: 1.817e-02
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Napomena: mali projektovani gradijent nije potvrdjen.
VALIDACIJA prozor=1, L2=0.01
Srednji uslovni NLL: 18.158383197
Srednji broj pogodaka: 1.293065
Ciljevi sa nultom masom: 0
Broj kola po pogocima 0..7: [85, 191, 128, 41, 2, 0, 0, 0]
VALIDACIJA prozor=3, L2=0.001
Trening prelaza: 2084
Broj karakteristika sa interceptom: 24
L2: 0.001
Deterministicki pocetak 1/2
Procena 10, najbolji NLL+L2=23.632928898
Procena 20, najbolji NLL+L2=20.281418577
Procena 30, najbolji NLL+L2=19.119562270
Procena 40, najbolji NLL+L2=18.721700214
Procena 50, najbolji NLL+L2=18.527654535
Procena 60, najbolji NLL+L2=18.436461075
Procena 70, najbolji NLL+L2=18.366114315
Procena 80, najbolji NLL+L2=18.317734275
Procena 90, najbolji NLL+L2=18.303350460
Procena 100, najbolji NLL+L2=18.289023194
Procena 110, najbolji NLL+L2=18.268790798
Procena 120, najbolji NLL+L2=18.258140026
Procena 130, najbolji NLL+L2=18.248289910
Procena 140, najbolji NLL+L2=18.242690564
Procena 150, najbolji NLL+L2=18.236399620
Procena 160, najbolji NLL+L2=18.231678080
Procena 170, najbolji NLL+L2=18.224225959
Procena 180, najbolji NLL+L2=18.216910997
Procena 190, najbolji NLL+L2=18.205543511
Procena 200, najbolji NLL+L2=18.199591445
Sacuvani NLL+L2: 18.197098559
Max|projektovani gradijent|: 4.850e-03
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Napomena: mali projektovani gradijent nije potvrdjen.
Deterministicki pocetak 2/2
Procena 10, najbolji NLL+L2=20.547300509
Procena 20, najbolji NLL+L2=19.410867316
Procena 30, najbolji NLL+L2=18.950031717
Procena 40, najbolji NLL+L2=18.302924622
Procena 50, najbolji NLL+L2=18.010660910
Procena 60, najbolji NLL+L2=17.904134552
Procena 70, najbolji NLL+L2=17.824899350
Procena 80, najbolji NLL+L2=17.781092286
Procena 90, najbolji NLL+L2=17.750152798
Procena 100, najbolji NLL+L2=17.728561490
Procena 110, najbolji NLL+L2=17.715269622
Procena 120, najbolji NLL+L2=17.706470614
Procena 130, najbolji NLL+L2=17.701483590
Procena 140, najbolji NLL+L2=17.695714884
Procena 150, najbolji NLL+L2=17.692304340
Procena 160, najbolji NLL+L2=17.689982846
Procena 170, najbolji NLL+L2=17.684137057
Procena 180, najbolji NLL+L2=17.677229950
Procena 190, najbolji NLL+L2=17.670330317
Procena 200, najbolji NLL+L2=17.665677975
Procena 210, najbolji NLL+L2=17.658920636
Procena 220, najbolji NLL+L2=17.646485174
Sacuvani NLL+L2: 17.645805807
Max|projektovani gradijent|: 1.747e-02
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Napomena: mali projektovani gradijent nije potvrdjen.
VALIDACIJA prozor=3, L2=0.001
Srednji uslovni NLL: 18.422941305
Srednji broj pogodaka: 1.250559
Ciljevi sa nultom masom: 0
Broj kola po pogocima 0..7: [88, 201, 118, 38, 2, 0, 0, 0]
VALIDACIJA prozor=3, L2=0.01
Trening prelaza: 2084
Broj karakteristika sa interceptom: 24
L2: 0.01
Deterministicki pocetak 1/2
Deterministicki pocetak 1/2
Procena 10, najbolji NLL+L2=23.633315431
Procena 20, najbolji NLL+L2=20.299073192
Procena 30, najbolji NLL+L2=19.157547935
Procena 40, najbolji NLL+L2=18.679387271
Procena 50, najbolji NLL+L2=18.521152128
Procena 60, najbolji NLL+L2=18.445623872
Procena 70, najbolji NLL+L2=18.375070812
Procena 80, najbolji NLL+L2=18.318463734
Procena 90, najbolji NLL+L2=18.276480860
Procena 100, najbolji NLL+L2=18.256490461
Procena 110, najbolji NLL+L2=18.237911685
Procena 120, najbolji NLL+L2=18.228774750
Procena 130, najbolji NLL+L2=18.226454282
Procena 140, najbolji NLL+L2=18.224673999
Procena 150, najbolji NLL+L2=18.223415888
Procena 160, najbolji NLL+L2=18.220864926
Procena 170, najbolji NLL+L2=18.219443867
Procena 180, najbolji NLL+L2=18.219169445
Procena 190, najbolji NLL+L2=18.219011616
Procena 200, najbolji NLL+L2=18.218894114
Sacuvani NLL+L2: 18.218860759
Max|projektovani gradijent|: 2.773e-04
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Napomena: mali projektovani gradijent nije potvrdjen.
Deterministicki pocetak 2/2
Procena 10, najbolji NLL+L2=20.581546043
Procena 20, najbolji NLL+L2=19.957163790
Procena 30, najbolji NLL+L2=19.887664903
Procena 40, najbolji NLL+L2=19.174017791
Procena 50, najbolji NLL+L2=18.542964041
Procena 60, najbolji NLL+L2=18.246934733
Procena 70, najbolji NLL+L2=18.098670173
Procena 80, najbolji NLL+L2=18.027562044
Procena 90, najbolji NLL+L2=17.996052674
Procena 100, najbolji NLL+L2=17.974029920
Procena 110, najbolji NLL+L2=17.951396822
Procena 120, najbolji NLL+L2=17.932874155
Procena 130, najbolji NLL+L2=17.917424860
Procena 140, najbolji NLL+L2=17.904330911
Procena 150, najbolji NLL+L2=17.898259093
Procena 160, najbolji NLL+L2=17.894380031
Procena 170, najbolji NLL+L2=17.888656788
Procena 180, najbolji NLL+L2=17.885174846
Procena 190, najbolji NLL+L2=17.884095681
Procena 200, najbolji NLL+L2=17.880568737
Procena 210, najbolji NLL+L2=17.876872569
Procena 220, najbolji NLL+L2=17.874167948
Sacuvani NLL+L2: 17.873525177
Max|projektovani gradijent|: 3.540e-03
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Napomena: mali projektovani gradijent nije potvrdjen.
VALIDACIJA prozor=3, L2=0.01
Srednji uslovni NLL: 18.229813978
Srednji broj pogodaka: 1.181208
Ciljevi sa nultom masom: 0
Broj kola po pogocima 0..7: [118, 171, 120, 35, 3, 0, 0, 0]

Izabrani prozor: 1
Izabrana regularizacija: 0.001
UCENJE ZA ODVOJENI TEST
Trening prelaza: 2531
Broj karakteristika sa interceptom: 8
L2: 0.001
Deterministicki pocetak 1/2
Procena 10, najbolji NLL+L2=23.718688833
Procena 20, najbolji NLL+L2=20.976929179
Procena 30, najbolji NLL+L2=19.387563360
Procena 40, najbolji NLL+L2=18.317146068
Procena 50, najbolji NLL+L2=18.150173176
Procena 60, najbolji NLL+L2=18.027724122
Procena 70, najbolji NLL+L2=17.971517022
Procena 80, najbolji NLL+L2=17.939022039
Procena 90, najbolji NLL+L2=17.922783489
Procena 100, najbolji NLL+L2=17.915598008
Procena 110, najbolji NLL+L2=17.910513380
Procena 120, najbolji NLL+L2=17.903720829
Procena 130, najbolji NLL+L2=17.899113208
Procena 140, najbolji NLL+L2=17.894274141
Procena 150, najbolji NLL+L2=17.892808684
Procena 160, najbolji NLL+L2=17.891798105
Procena 170, najbolji NLL+L2=17.888683700
Procena 180, najbolji NLL+L2=17.886318311
Procena 190, najbolji NLL+L2=17.885885975
Procena 200, najbolji NLL+L2=17.885602967
Sacuvani NLL+L2: 17.885445708
Max|projektovani gradijent|: 3.352e-03
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Napomena: mali projektovani gradijent nije potvrdjen.
Deterministicki pocetak 2/2
Procena 10, najbolji NLL+L2=20.042741886
Procena 20, najbolji NLL+L2=19.118549870
Procena 30, najbolji NLL+L2=18.709571070
Procena 40, najbolji NLL+L2=18.374739349
Procena 50, najbolji NLL+L2=18.269621969
Procena 60, najbolji NLL+L2=18.221815400
Procena 70, najbolji NLL+L2=18.163596432
Procena 80, najbolji NLL+L2=18.141009614
Procena 90, najbolji NLL+L2=18.129578583
Procena 100, najbolji NLL+L2=18.124240297
Procena 110, najbolji NLL+L2=18.122651133
Procena 120, najbolji NLL+L2=18.121178035
Procena 130, najbolji NLL+L2=18.119491332
Procena 140, najbolji NLL+L2=18.100679959
Procena 150, najbolji NLL+L2=18.086855439
Procena 160, najbolji NLL+L2=18.081326944
Procena 170, najbolji NLL+L2=18.078551655
Procena 180, najbolji NLL+L2=18.077028512
Procena 190, najbolji NLL+L2=18.073852535
Procena 200, najbolji NLL+L2=18.071543679
Sacuvani NLL+L2: 18.070899995
Max|projektovani gradijent|: 9.351e-03
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Napomena: mali projektovani gradijent nije potvrdjen.
ODVOJENI HRONOLOSKI TEST
Srednji uslovni NLL: 18.386437785
Srednji broj pogodaka: 1.322148
Ciljevi sa nultom masom: 0
Broj kola po pogocima 0..7: [100, 164, 132, 43, 6, 2, 0, 0]
ZAVRSNO UCENJE NA CELOJ ISTORIJI
Trening prelaza: 2978
Broj karakteristika sa interceptom: 8
L2: 0.001
Deterministicki pocetak 1/2
Procena 10, najbolji NLL+L2=23.616614736
Procena 20, najbolji NLL+L2=20.590436526
Procena 30, najbolji NLL+L2=18.974872162
Procena 40, najbolji NLL+L2=18.658814173
Procena 50, najbolji NLL+L2=18.564478111
Procena 60, najbolji NLL+L2=18.527294944
Procena 70, najbolji NLL+L2=18.504050498
Procena 80, najbolji NLL+L2=18.486083027
Procena 90, najbolji NLL+L2=18.474541581
Procena 100, najbolji NLL+L2=18.465774927
Procena 110, najbolji NLL+L2=18.457366340
Procena 120, najbolji NLL+L2=18.437738625
Procena 130, najbolji NLL+L2=18.425844591
Procena 140, najbolji NLL+L2=18.421474720
Procena 150, najbolji NLL+L2=18.419392169
Procena 160, najbolji NLL+L2=18.416921516
Procena 170, najbolji NLL+L2=18.414252447
Procena 180, najbolji NLL+L2=18.407837346
Procena 190, najbolji NLL+L2=18.402870733
Procena 200, najbolji NLL+L2=18.400865338
Sacuvani NLL+L2: 18.399931627
Max|projektovani gradijent|: 9.577e-03
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Napomena: mali projektovani gradijent nije potvrdjen.
Deterministicki pocetak 2/2
Procena 10, najbolji NLL+L2=19.757069033
Procena 20, najbolji NLL+L2=19.065785237
Procena 30, najbolji NLL+L2=18.696327026
Procena 40, najbolji NLL+L2=18.441997306
Procena 50, najbolji NLL+L2=18.363737564
Procena 60, najbolji NLL+L2=18.286308233
Procena 70, najbolji NLL+L2=18.236151138
Procena 80, najbolji NLL+L2=18.177922889
Procena 90, najbolji NLL+L2=18.135017368
Procena 100, najbolji NLL+L2=18.079189330
Procena 110, najbolji NLL+L2=18.051353225
Procena 120, najbolji NLL+L2=18.012746054
Procena 130, najbolji NLL+L2=17.989576610
Procena 140, najbolji NLL+L2=17.977179616
Procena 150, najbolji NLL+L2=17.970155962
Procena 160, najbolji NLL+L2=17.960500959
Procena 170, najbolji NLL+L2=17.947019658
Procena 180, najbolji NLL+L2=17.941315142
Procena 190, najbolji NLL+L2=17.939778047
Procena 200, najbolji NLL+L2=17.938382057
Procena 210, najbolji NLL+L2=17.937481688
Sacuvani NLL+L2: 17.937422270
Max|projektovani gradijent|: 1.254e-02
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Napomena: mali projektovani gradijent nije potvrdjen.

NAUCENI DIRACOVI PARAMETRI I SPINOR
Registar 1, c=12.960828, mu=9.395000
Spinor 0 masa=0.50539036, spinor 1 masa=0.49460964, cistoca=0.50749698
Registar 2, c=13.765585, mu=12.793743
Spinor 0 masa=0.69354202, spinor 1 masa=0.30645798, cistoca=0.57522403
Registar 3, c=6.802730, mu=3.905361
Spinor 0 masa=0.61727870, spinor 1 masa=0.38272130, cistoca=0.62406703
Registar 4, c=11.064458, mu=8.036233
Spinor 0 masa=0.45237062, spinor 1 masa=0.54762938, cistoca=0.53697816
Registar 5, c=12.879143, mu=9.950593
Spinor 0 masa=0.51118870, spinor 1 masa=0.48881130, cistoca=0.50043759
Registar 6, c=10.224555, mu=11.158533
Spinor 0 masa=0.53201938, spinor 1 masa=0.46798062, cistoca=0.51256343
Registar 7, c=14.304780, mu=8.499318
Spinor 0 masa=0.64990002, spinor 1 masa=0.35009998, cistoca=0.59873730

Log legalne mase: -8.755630048
Log uslovne mase NEXT: -9.673902287
Promena kroz stvarne istorijske kontekste: 2.408378649e-01
Uticaj istorijskih spinorskih ulaza na marginale: 1.734084194e-01
Raspon NEXT kombinacije: 38
Udeo istorijskih raspona <= NEXT raspon: 100.000%

========================================================================
loto7_4716_k89_loto_plus_1736.csv
Stvarni broj kola: 1735
SHA256: 56225ed0f0be7e2d3e30779804d8fa45e96ecfb9d14441eb2a731dd7fff38ae6
Trening prelaza: 1213
Validacionih prelaza: 260
Test prelaza: 261
VALIDACIJA prozor=1, L2=0.001
Trening prelaza: 1213
Broj karakteristika sa interceptom: 8
L2: 0.001
Deterministicki pocetak 1/2
Procena 10, najbolji NLL+L2=23.717088204
Procena 20, najbolji NLL+L2=20.634743656
Procena 30, najbolji NLL+L2=19.500430393
Procena 40, najbolji NLL+L2=18.716221422
Procena 50, najbolji NLL+L2=18.200332427
Procena 60, najbolji NLL+L2=17.904189512
Procena 70, najbolji NLL+L2=17.805610481
Procena 80, najbolji NLL+L2=17.752978393
Procena 90, najbolji NLL+L2=17.722335293
Procena 100, najbolji NLL+L2=17.702925324
Procena 110, najbolji NLL+L2=17.695459918
Procena 120, najbolji NLL+L2=17.690027559
Procena 130, najbolji NLL+L2=17.684663859
Procena 140, najbolji NLL+L2=17.683338659
Procena 150, najbolji NLL+L2=17.683054364
Procena 160, najbolji NLL+L2=17.682850830
Procena 170, najbolji NLL+L2=17.682785902
Procena 180, najbolji NLL+L2=17.682772868
Procena 190, najbolji NLL+L2=17.682769550
Procena 200, najbolji NLL+L2=17.682768865
Procena 210, najbolji NLL+L2=17.682768693
Sacuvani NLL+L2: 17.682768614
Max|projektovani gradijent|: 3.512e-05
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Napomena: mali projektovani gradijent nije potvrdjen.
Deterministicki pocetak 2/2
Procena 10, najbolji NLL+L2=19.588189800
Procena 20, najbolji NLL+L2=19.057568547
Procena 30, najbolji NLL+L2=18.522761045
Procena 40, najbolji NLL+L2=18.232293817
Procena 50, najbolji NLL+L2=18.099506958
Procena 60, najbolji NLL+L2=17.967095972
Procena 70, najbolji NLL+L2=17.882309847
Procena 80, najbolji NLL+L2=17.838634459
Procena 90, najbolji NLL+L2=17.796953269
Procena 100, najbolji NLL+L2=17.774601499
Procena 110, najbolji NLL+L2=17.751713733
Procena 120, najbolji NLL+L2=17.727135460
Procena 130, najbolji NLL+L2=17.713129420
Procena 140, najbolji NLL+L2=17.702665245
Procena 150, najbolji NLL+L2=17.684152349
Procena 160, najbolji NLL+L2=17.667275634
Procena 170, najbolji NLL+L2=17.662988230
Procena 180, najbolji NLL+L2=17.661998407
Procena 190, najbolji NLL+L2=17.661004478
Procena 200, najbolji NLL+L2=17.659370910
Procena 210, najbolji NLL+L2=17.653460174
Sacuvani NLL+L2: 17.648385082
Max|projektovani gradijent|: 5.032e-02
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Napomena: mali projektovani gradijent nije potvrdjen.
VALIDACIJA prozor=1, L2=0.001
Srednji uslovni NLL: 17.927537598
Srednji broj pogodaka: 1.211538
Ciljevi sa nultom masom: 0
Broj kola po pogocima 0..7: [61, 106, 71, 21, 1, 0, 0, 0]
VALIDACIJA prozor=1, L2=0.01
Trening prelaza: 1213
Broj karakteristika sa interceptom: 8
L2: 0.01
Deterministicki pocetak 1/2
Procena 10, najbolji NLL+L2=23.718410366
Procena 20, najbolji NLL+L2=20.649475654
Procena 30, najbolji NLL+L2=19.760864418
Procena 40, najbolji NLL+L2=18.810766048
Procena 50, najbolji NLL+L2=18.515739389
Procena 60, najbolji NLL+L2=18.444493958
Procena 70, najbolji NLL+L2=18.396779751
Procena 80, najbolji NLL+L2=18.354550688
Procena 90, najbolji NLL+L2=18.329413452
Procena 100, najbolji NLL+L2=18.318971994
Procena 110, najbolji NLL+L2=18.310478491
Procena 120, najbolji NLL+L2=18.307855646
Procena 130, najbolji NLL+L2=18.305359768
Procena 140, najbolji NLL+L2=18.301415332
Procena 150, najbolji NLL+L2=18.295100009
Procena 160, najbolji NLL+L2=18.288891118
Procena 170, najbolji NLL+L2=18.284877785
Procena 180, najbolji NLL+L2=18.277478095
Procena 190, najbolji NLL+L2=18.272793065
Procena 200, najbolji NLL+L2=18.263187592
Procena 210, najbolji NLL+L2=18.261032731
Sacuvani NLL+L2: 18.259478133
Max|projektovani gradijent|: 6.656e-03
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Napomena: mali projektovani gradijent nije potvrdjen.
Deterministicki pocetak 2/2
Procena 10, najbolji NLL+L2=19.595458963
Procena 20, najbolji NLL+L2=19.066133797
Procena 30, najbolji NLL+L2=18.549077195
Procena 40, najbolji NLL+L2=18.310058667
Procena 50, najbolji NLL+L2=18.095824405
Procena 60, najbolji NLL+L2=17.949759004
Procena 70, najbolji NLL+L2=17.868481709
Procena 80, najbolji NLL+L2=17.842471393
Procena 90, najbolji NLL+L2=17.800256596
Procena 100, najbolji NLL+L2=17.776270269
Procena 110, najbolji NLL+L2=17.764114609
Procena 120, najbolji NLL+L2=17.751950129
Procena 130, najbolji NLL+L2=17.743728691
Procena 140, najbolji NLL+L2=17.736244199
Procena 150, najbolji NLL+L2=17.730838586
Procena 160, najbolji NLL+L2=17.728339028
Procena 170, najbolji NLL+L2=17.727144294
Procena 180, najbolji NLL+L2=17.726654791
Procena 190, najbolji NLL+L2=17.726211930
Procena 200, najbolji NLL+L2=17.725516537
Procena 210, najbolji NLL+L2=17.724787803
Sacuvani NLL+L2: 17.724787803
Max|projektovani gradijent|: 1.730e-03
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Napomena: mali projektovani gradijent nije potvrdjen.
VALIDACIJA prozor=1, L2=0.01
Srednji uslovni NLL: 18.031221375
Srednji broj pogodaka: 1.300000
Ciljevi sa nultom masom: 0
Broj kola po pogocima 0..7: [48, 120, 62, 26, 4, 0, 0, 0]
VALIDACIJA prozor=3, L2=0.001
Trening prelaza: 1213
Broj karakteristika sa interceptom: 24
L2: 0.001
Deterministicki pocetak 1/2
Procena 10, najbolji NLL+L2=24.064076773
Procena 20, najbolji NLL+L2=20.514072644
Procena 30, najbolji NLL+L2=18.741446664
Procena 40, najbolji NLL+L2=18.427528469
Procena 50, najbolji NLL+L2=18.292404765
Procena 60, najbolji NLL+L2=18.207590169
Procena 70, najbolji NLL+L2=18.143812550
Procena 80, najbolji NLL+L2=18.110890092
Procena 90, najbolji NLL+L2=18.084733601
Procena 100, najbolji NLL+L2=18.054213356
Procena 110, najbolji NLL+L2=18.030500297
Procena 120, najbolji NLL+L2=18.021626484
Procena 130, najbolji NLL+L2=18.012853302
Procena 140, najbolji NLL+L2=18.005744474
Procena 150, najbolji NLL+L2=17.998338087
Procena 160, najbolji NLL+L2=17.988564502
Procena 170, najbolji NLL+L2=17.984908263
Procena 180, najbolji NLL+L2=17.983215381
Procena 190, najbolji NLL+L2=17.982078072
Procena 200, najbolji NLL+L2=17.972933776
Sacuvani NLL+L2: 17.962478855
Max|projektovani gradijent|: 1.418e-02
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Napomena: mali projektovani gradijent nije potvrdjen.
Deterministicki pocetak 2/2
Procena 10, najbolji NLL+L2=20.685329685
Procena 20, najbolji NLL+L2=18.502164454
Procena 30, najbolji NLL+L2=18.050266925
Procena 40, najbolji NLL+L2=17.912834255
Procena 50, najbolji NLL+L2=17.788853082
Procena 60, najbolji NLL+L2=17.714260161
Procena 70, najbolji NLL+L2=17.611766005
Procena 80, najbolji NLL+L2=17.522053720
Procena 90, najbolji NLL+L2=17.467687837
Procena 100, najbolji NLL+L2=17.433361330
Procena 110, najbolji NLL+L2=17.422074134
Procena 120, najbolji NLL+L2=17.404214207
Procena 130, najbolji NLL+L2=17.380253914
Procena 140, najbolji NLL+L2=17.348148458
Procena 150, najbolji NLL+L2=17.334910916
Procena 160, najbolji NLL+L2=17.327333070
Procena 170, najbolji NLL+L2=17.320912743
Procena 180, najbolji NLL+L2=17.317328814
Procena 190, najbolji NLL+L2=17.315279353
Procena 200, najbolji NLL+L2=17.313513007
Sacuvani NLL+L2: 17.313071359
Max|projektovani gradijent|: 2.777e-03
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Napomena: mali projektovani gradijent nije potvrdjen.
VALIDACIJA prozor=3, L2=0.001
Srednji uslovni NLL: 18.012690594
Srednji broj pogodaka: 1.226923
Ciljevi sa nultom masom: 0
Broj kola po pogocima 0..7: [58, 109, 71, 20, 2, 0, 0, 0]
VALIDACIJA prozor=3, L2=0.01
Trening prelaza: 1213
Broj karakteristika sa interceptom: 24
L2: 0.01
Deterministicki pocetak 1/2
Procena 10, najbolji NLL+L2=24.064317916
Procena 20, najbolji NLL+L2=20.551366413
Procena 30, najbolji NLL+L2=18.784776391
Procena 40, najbolji NLL+L2=18.422727905
Procena 50, najbolji NLL+L2=18.216350079
Procena 60, najbolji NLL+L2=18.134501281
Procena 70, najbolji NLL+L2=18.064594333
Procena 80, najbolji NLL+L2=18.022227197
Procena 90, najbolji NLL+L2=17.991660194
Procena 100, najbolji NLL+L2=17.960316177
Procena 110, najbolji NLL+L2=17.949048066
Procena 120, najbolji NLL+L2=17.943139940
Procena 130, najbolji NLL+L2=17.938370641
Procena 140, najbolji NLL+L2=17.928735286
Procena 150, najbolji NLL+L2=17.908230431
Procena 160, najbolji NLL+L2=17.860474364
Procena 170, najbolji NLL+L2=17.849745246
Procena 180, najbolji NLL+L2=17.835415898
Procena 190, najbolji NLL+L2=17.829256606
Procena 200, najbolji NLL+L2=17.822159611
Sacuvani NLL+L2: 17.817636997
Max|projektovani gradijent|: 4.134e-02
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Napomena: mali projektovani gradijent nije potvrdjen.
Deterministicki pocetak 2/2
Procena 10, najbolji NLL+L2=20.686168463
Procena 20, najbolji NLL+L2=18.519196255
Procena 30, najbolji NLL+L2=18.060055295
Procena 40, najbolji NLL+L2=17.917335265
Procena 50, najbolji NLL+L2=17.796987074
Procena 60, najbolji NLL+L2=17.728681921
Procena 70, najbolji NLL+L2=17.597199384
Procena 80, najbolji NLL+L2=17.530894786
Procena 90, najbolji NLL+L2=17.488209317
Procena 100, najbolji NLL+L2=17.454913186
Procena 110, najbolji NLL+L2=17.429706129
Procena 120, najbolji NLL+L2=17.412425400
Procena 130, najbolji NLL+L2=17.395662486
Procena 140, najbolji NLL+L2=17.371585173
Procena 150, najbolji NLL+L2=17.348318034
Procena 160, najbolji NLL+L2=17.341037574
Procena 170, najbolji NLL+L2=17.336651524
Procena 180, najbolji NLL+L2=17.334843178
Procena 190, najbolji NLL+L2=17.333988993
Procena 200, najbolji NLL+L2=17.333562650
Sacuvani NLL+L2: 17.333477120
Max|projektovani gradijent|: 5.618e-03
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Napomena: mali projektovani gradijent nije potvrdjen.
VALIDACIJA prozor=3, L2=0.01
Srednji uslovni NLL: 18.200559492
Srednji broj pogodaka: 1.330769
Ciljevi sa nultom masom: 0
Broj kola po pogocima 0..7: [50, 110, 71, 23, 5, 1, 0, 0]

Izabrani prozor: 1
Izabrana regularizacija: 0.001
UCENJE ZA ODVOJENI TEST
Trening prelaza: 1473
Broj karakteristika sa interceptom: 8
L2: 0.001
Deterministicki pocetak 1/2
Procena 10, najbolji NLL+L2=23.778432725
Procena 20, najbolji NLL+L2=20.291788614
Procena 30, najbolji NLL+L2=19.684419466
Procena 40, najbolji NLL+L2=18.920196774
Procena 50, najbolji NLL+L2=18.651630205
Procena 60, najbolji NLL+L2=18.414664575
Procena 70, najbolji NLL+L2=18.293101745
Procena 80, najbolji NLL+L2=18.243258472
Procena 90, najbolji NLL+L2=18.214619364
Procena 100, najbolji NLL+L2=18.204634422
Procena 110, najbolji NLL+L2=18.188593147
Procena 120, najbolji NLL+L2=18.165978863
Procena 130, najbolji NLL+L2=18.150703215
Procena 140, najbolji NLL+L2=18.132290753
Procena 150, najbolji NLL+L2=18.118787613
Procena 160, najbolji NLL+L2=18.116639104
Procena 170, najbolji NLL+L2=18.114055475
Procena 180, najbolji NLL+L2=18.110241485
Procena 190, najbolji NLL+L2=18.104926817
Procena 200, najbolji NLL+L2=18.101466385
Procena 210, najbolji NLL+L2=18.098142823
Procena 220, najbolji NLL+L2=18.097314162
Sacuvani NLL+L2: 18.097314162
Max|projektovani gradijent|: 1.262e-03
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Napomena: mali projektovani gradijent nije potvrdjen.
Deterministicki pocetak 2/2
Procena 10, najbolji NLL+L2=19.491364810
Procena 20, najbolji NLL+L2=18.615076548
Procena 30, najbolji NLL+L2=18.429231644
Procena 40, najbolji NLL+L2=18.284506745
Procena 50, najbolji NLL+L2=18.202069633
Procena 60, najbolji NLL+L2=18.071051079
Procena 70, najbolji NLL+L2=17.913653539
Procena 80, najbolji NLL+L2=17.844172985
Procena 90, najbolji NLL+L2=17.814980959
Procena 100, najbolji NLL+L2=17.786846435
Procena 110, najbolji NLL+L2=17.721246528
Procena 120, najbolji NLL+L2=17.663079372
Procena 130, najbolji NLL+L2=17.647464971
Procena 140, najbolji NLL+L2=17.639185541
Procena 150, najbolji NLL+L2=17.629951462
Procena 160, najbolji NLL+L2=17.617658771
Procena 170, najbolji NLL+L2=17.609694393
Procena 180, najbolji NLL+L2=17.604784903
Procena 190, najbolji NLL+L2=17.595711240
Procena 200, najbolji NLL+L2=17.578197994
Procena 210, najbolji NLL+L2=17.573132256
Sacuvani NLL+L2: 17.571298317
Max|projektovani gradijent|: 4.746e-03
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Napomena: mali projektovani gradijent nije potvrdjen.
ODVOJENI HRONOLOSKI TEST
Srednji uslovni NLL: 17.846045853
Srednji broj pogodaka: 1.306513
Ciljevi sa nultom masom: 0
Broj kola po pogocima 0..7: [56, 101, 74, 28, 2, 0, 0, 0]
ZAVRSNO UCENJE NA CELOJ ISTORIJI
Trening prelaza: 1734
Broj karakteristika sa interceptom: 8
L2: 0.001
Deterministicki pocetak 1/2
Procena 10, najbolji NLL+L2=23.635846750
Procena 20, najbolji NLL+L2=19.589202902
Procena 30, najbolji NLL+L2=18.995438904
Procena 40, najbolji NLL+L2=18.872522556
Procena 50, najbolji NLL+L2=18.775006393
Procena 60, najbolji NLL+L2=18.718259636
Procena 70, najbolji NLL+L2=18.672803643
Procena 80, najbolji NLL+L2=18.626860507
Procena 90, najbolji NLL+L2=18.570285011
Procena 100, najbolji NLL+L2=18.546387536
Procena 110, najbolji NLL+L2=18.526832816
Procena 120, najbolji NLL+L2=18.510359473
Procena 130, najbolji NLL+L2=18.508149897
Procena 140, najbolji NLL+L2=18.506512547
Procena 150, najbolji NLL+L2=18.505969025
Procena 160, najbolji NLL+L2=18.505612109
Procena 170, najbolji NLL+L2=18.505448743
Procena 180, najbolji NLL+L2=18.505381507
Procena 190, najbolji NLL+L2=18.504716610
Procena 200, najbolji NLL+L2=18.504327958
Sacuvani NLL+L2: 18.504271164
Max|projektovani gradijent|: 5.844e-04
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Napomena: mali projektovani gradijent nije potvrdjen.
Deterministicki pocetak 2/2
Procena 10, najbolji NLL+L2=19.314994676
Procena 20, najbolji NLL+L2=18.872805374
Procena 30, najbolji NLL+L2=18.599004819
Procena 40, najbolji NLL+L2=18.392639604
Procena 50, najbolji NLL+L2=18.242854078
Procena 60, najbolji NLL+L2=18.137880282
Procena 70, najbolji NLL+L2=18.008786943
Procena 80, najbolji NLL+L2=17.953441232
Procena 90, najbolji NLL+L2=17.895714057
Procena 100, najbolji NLL+L2=17.835968500
Procena 110, najbolji NLL+L2=17.798690970
Procena 120, najbolji NLL+L2=17.780420421
Procena 130, najbolji NLL+L2=17.766457385
Procena 140, najbolji NLL+L2=17.751615678
Procena 150, najbolji NLL+L2=17.742175582
Procena 160, najbolji NLL+L2=17.728983465
Procena 170, najbolji NLL+L2=17.697673807
Procena 180, najbolji NLL+L2=17.680256418
Procena 190, najbolji NLL+L2=17.674759741
Procena 200, najbolji NLL+L2=17.671097427
Sacuvani NLL+L2: 17.666585091
Max|projektovani gradijent|: 8.205e-03
Status: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
Napomena: mali projektovani gradijent nije potvrdjen.

NAUCENI DIRACOVI PARAMETRI I SPINOR
Registar 1, c=15.143100, mu=12.482367
Spinor 0 masa=0.37621981, spinor 1 masa=0.62378019, cistoca=0.55212013
Registar 2, c=17.117223, mu=13.967736
Spinor 0 masa=0.55108904, spinor 1 masa=0.44891096, cistoca=0.63782978
Registar 3, c=7.697430, mu=6.423251
Spinor 0 masa=0.48086772, spinor 1 masa=0.51913228, cistoca=0.52329176
Registar 4, c=12.139255, mu=8.188101
Spinor 0 masa=0.64319813, spinor 1 masa=0.35680187, cistoca=0.65063236
Registar 5, c=11.766182, mu=13.251942
Spinor 0 masa=0.22707256, spinor 1 masa=0.77292744, cistoca=0.64921786
Registar 6, c=11.474979, mu=11.113647
Spinor 0 masa=0.50145827, spinor 1 masa=0.49854173, cistoca=0.57087852
Registar 7, c=14.227590, mu=9.536252
Spinor 0 masa=0.70381763, spinor 1 masa=0.29618237, cistoca=0.58611768

Log legalne mase: -10.028627982
Log uslovne mase NEXT: -10.401041206
Promena kroz stvarne istorijske kontekste: 2.570724057e-01
Uticaj istorijskih spinorskih ulaza na marginale: 1.917775744e-01
Raspon NEXT kombinacije: 36
Udeo istorijskih raspona <= NEXT raspon: 92.450%


========================================================================
POREDJENJE NEZAVISNIH REZULTATA

========================================================================

loto7_4716_k89.csv NEXT: 19 21 22 23 24 25 38

loto7_4716_k89_loto_2980.csv NEXT: 1 2 12 27 37 38 39

loto7_4716_k89_loto_plus_1736.csv NEXT: 2 4 10 11 26 36 38
"""



"""
Osnovna ideja jeste evolucija spregnutih komponenti spinora preko Diracovog Hamiltonijana. 
Postoje kvantne simulacije te jednačine i njene realizacije kvantnim hodom. 

- Šest kubita za brojna stanja i jedan dodatni spinorski kubit po registru.
- Dve kompleksne amplitude po brojnom stanju, čije sprezanje i interferencija određuju izlaznu raspodelu.
- Istorija kola određuje početni spinor, a parametri evolucije uče se regresijom raspodele.
- Kinetički član i član mase ostaju u Diracovoj strukturi, sa odgovarajućim antikomutirajućim matricama. Masa bi bila parametar modela.
- Verovatnoća broja dobija se sabiranjem kvadrata modula obe spinorske komponente.

Sprezanje spinorskih komponenti i Diracova evolucija, uz hronološku validaciju i zasebnu obradu sva tri CSV-a.

Diracov kinetički član, član mase i dve spregnute komponente spinora. 
Svaki registar koristi šest kubita za brojna stanja i jedan spinorski kubit.

Koristim spektralnu diskretizaciju Diracovog Hamiltonijana na periodičnoj mreži od 39 stanja. 
Iz istorije se uči početni spinor, masa, brzina prostiranja i potencijal. 
Obuka neposredno koristi zajedničku raspodelu legalnih kombinacija, a izlaznu verovatnoću broja dobijam sabiranjem obe spinorske komponente.

Diracov Hamiltonijan sa kinetičkim članom, masenim članom i potencijalom. 
Svaki od sedam registara ima šest prostornih kubita i jedan spinorski kubit. 
Verovatnoća broja dobija se iz obe spinorske komponente. Osnova kvantne simulacije. 

Parametri početnog spinora i Hamiltonijana uče se neposredno kroz zajedničku raspodelu legalnih kombinacija. 

Poređenje logaritama zamenio sam poređenjem Bornovih verovatnoća uz zasebnu proveru normalizacije.
"""
