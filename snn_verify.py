#!/usr/bin/env python3
"""Verify infeasibility certificates for Laplacian spectrum S_{n,n}.

Originally created by ChatGPT-6 Astra for Nathaniel Johnston
(njohnston@mta.ca) as companion code for the paper "The Laplacian S_{n,n}
conjecture is true".

Python 3.10 or later, no other dependencies.

Usage:
    python snn_verify.py n24

The argument is a directory containing the proof/certificates. Its run.json
describes the order and task partition; parts/ contains the witness streams,
and certificates/ contains the integer certificates.

A graph with spectrum {0, 1, ..., n-1} has degrees between 2 and n-3, with
sum d = n(n-1)/2 and sum d^2 = n(n-1)(n-2)/3. Complementation preserves the
spectrum and reverses the degree multiplicities. The verifier enumerates
every sequence satisfying these necessary conditions and checks one
representative of each complementary pair.

This verifier checks that each representative has an integer certificate h
with h^T A >= 0 and h^T b < 0 for its degree sequence or its complement.
Here A and b encode the nine families of linear systems from the
accompanying paper.

Exit codes: 0 = nonexistence verified; 1 = verification failed;
2 = invalid command-line arguments; 130 = interrupted.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import struct
import sys


DIRECTORY_FORMAT = "snn-search-directory-v1"
MODEL = "snn-local-fourth-moments-v1"


class VerificationError(ValueError):
    """A required proof record is missing, malformed, or mathematically invalid."""


def load_json(path: Path):
    """Read a UTF-8 JSON record."""
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def require_object(value, description: str) -> dict:
    if not isinstance(value, dict):
        raise VerificationError(f"{description} must be a JSON object")
    return value


def integer_field(record: dict, name: str, minimum: int = 0) -> int:
    value = record.get(name)
    # JSON booleans must not be accepted as integer coefficients or counts.
    if type(value) is not int or value < minimum:
        raise VerificationError(f"{name} must be an integer at least {minimum}")
    return value


def require_lp_order(n: int) -> None:
    if n < 5 or n % 4 not in (0, 1):
        raise VerificationError(f"Order {n} is not supported by this LP proof format")


@dataclass(frozen=True)
class Certificate:
    """A checked certificate and its sparse polynomial h^T b(m)."""

    constant: int
    linear: tuple[tuple[int, int], ...]
    pairs: tuple[tuple[int, int, int], ...]
    columns: int

    def evaluate(self, counts: tuple[int, ...]) -> int:
        value = self.constant + sum(h * counts[i] for i, h in self.linear)
        for i, j, h in self.pairs:
            pair_count = (counts[i] * (counts[i] - 1) // 2
                          if i == j else counts[i] * counts[j])
            value += h * pair_count
        return value


def check_certificate(n: int, coefficients) -> Certificate:
    """Check h^T A >= 0, then collect the coefficients of h^T b(m).

    Certificate rows are stored in this order: five moment equations for each
    degree 2,...,n-3; normalization for eigenvalues 1,...,n-1; unordered degree
    pairs in lexicographic order; degree counts; common-neighbor identities.
    The zeroth-moment equation is multiplied by n, so all entries are integers.
    """
    if not isinstance(coefficients, list) or any(type(h) is not int for h in coefficients):
        raise VerificationError("Certificate coefficients must be a list of integers")
    degrees = range(2, n - 2)
    pairs = [(c, d) for c in degrees for d in degrees if c <= d]
    size = len(degrees)
    expected = 7 * size + n - 1 + len(pairs)
    if len(coefficients) != expected:
        raise VerificationError(f"Certificate has {len(coefficients)} entries; expected {expected}")

    moments = {d: coefficients[5 * (d - 2):5 * (d - 1)] for d in degrees}
    offset = 5 * size
    spectral = dict(zip(range(1, n), coefficients[offset:offset + n - 1]))
    offset += n - 1
    pair_duals = dict(zip(pairs, coefficients[offset:offset + len(pairs)]))
    offset += len(pairs)
    degree_duals = dict(zip(degrees, coefficients[offset:offset + size]))
    common_duals = dict(zip(degrees, coefficients[offset + size:]))

    columns = 0
    for d in degrees:
        h0, h1, h2, h3, h4 = moments[d]
        for k in range(1, n):
            value = n * h0 + k * h1 + k**2 * h2 + k**3 * h3 + k**4 * h4 + spectral[k]
            if value < 0:
                raise VerificationError(f"Negative certificate column for y_({d},{k}): {value}")
            columns += 1

    for c, d in pairs:
        for a in (0, 1):
            lower = max(0, c + d - (n - 2 + 2 * a))
            upper = min(c, d) - a
            for b in range(lower, upper + 1):
                value = pair_duals[c, d]
                # An unordered pair contributes at both endpoints, including
                # twice in the same degree class when c == d.
                for degree, other in ((c, d), (d, c)):
                    value += a * degree_duals[degree]
                    value += (b - a * (other - 1)) * common_duals[degree]
                    value -= a * (other - b) * moments[degree][3]
                    value -= (b - (c + d) * a)**2 * moments[degree][4]
                if value < 0:
                    raise VerificationError(f"Negative certificate column for z_({a},{b},{c},{d}): {value}")
                columns += 1

    linear = []
    for d in degrees:
        constants = (n - 1, d, d**2 + d, d**3 + 2 * d**2, (d**2 + d)**2)
        value = sum(h * q for h, q in zip(moments[d], constants)) + d * degree_duals[d]
        if value:
            linear.append((d - 2, value))
    quadratic = tuple((c - 2, d - 2, h) for (c, d), h in pair_duals.items() if h)
    return Certificate(sum(spectral.values()), tuple(linear), quadratic, columns)


def can_complete(lower: int, upper: int, remaining: int, total: int, squares: int) -> bool:
    """Apply necessary bounds to remaining degrees in [lower, upper].

    For fixed total, balanced integer entries minimize the square sum.
    Entries at the endpoints, with at most one intermediate entry, maximize
    it. Also, a square has the same parity as its integer base.
    """
    if remaining == 0:
        return total == squares == 0
    if remaining < 0 or lower > upper or not remaining * lower <= total <= remaining * upper:
        return False
    if (squares - total) % 2:
        return False
    quotient, remainder = divmod(total, remaining)
    minimum = (remaining - remainder) * quotient**2 + remainder * (quotient + 1)**2
    if lower == upper:
        maximum = remaining * lower**2
    else:
        high_count, excess = divmod(total - remaining * lower, upper - lower)
        maximum = high_count * upper**2 + (remaining - high_count) * lower**2
        maximum += (lower + excess)**2 - lower**2
    return minimum <= squares <= maximum


def degree_vectors(n: int, prefix: tuple[int, ...] = (), length: int | None = None):
    """Generate degree multiplicities, or the required task prefixes.

    Descending multiplicities give ascending expanded degree sequences, the
    order used by the proof's witness stream. Stopping at a shorter length
    generates every prefix that passes the necessary residual bounds, even
    when that prefix ultimately has no complete degree sequence.
    """
    size = n - 4
    length = size if length is None else length
    if not 0 <= len(prefix) <= length <= size or any(type(m) is not int or m < 0 for m in prefix):
        raise VerificationError("Invalid degree-enumeration prefix or length")
    total = n * (n - 1) // 2 - sum((i + 2) * m for i, m in enumerate(prefix))
    squares = n * (n - 1) * (n - 2) // 3 - sum((i + 2)**2 * m for i, m in enumerate(prefix))
    upper = n - 3

    def visit(chosen, remaining, total, squares):
        lower = len(chosen) + 2
        if not can_complete(lower, upper, remaining, total, squares):
            return
        if len(chosen) == length:
            yield chosen
            return
        if remaining == 0:
            yield chosen + (0,) * (length - len(chosen))
            return
        if length == size and lower == upper:
            yield chosen + (remaining,)
            return
        if length == size and lower + 1 == upper:
            # Two remaining multiplicities are determined by their count and
            # sum; the square sum is checked before accepting the sequence.
            high_count = total - lower * remaining
            low_count = remaining - high_count
            if low_count >= 0 and high_count >= 0 and lower**2 * low_count + upper**2 * high_count == squares:
                yield chosen + (low_count, high_count)
            return
        for multiplicity in range(remaining, -1, -1):
            yield from visit(chosen + (multiplicity,), remaining - multiplicity,
                             total - lower * multiplicity, squares - lower**2 * multiplicity)

    yield from visit(prefix, n - sum(prefix), total, squares)


def expanded(counts: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(d for d, m in enumerate(counts, start=2) for _ in range(m))


def task_name(prefix: tuple[int, ...]) -> str:
    return "p_" + ("_".join(map(str, prefix)) if prefix else "all")


def require_digest(digest) -> None:
    if type(digest) is not str or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise VerificationError("Invalid certificate identifier")


@lru_cache(maxsize=None)
def load_certificate(work: Path, n: int, digest: str) -> Certificate:
    """Authenticate and check each referenced certificate once per run."""
    require_digest(digest)
    path = work / "certificates" / (digest + ".json")
    data = require_object(load_json(path), str(path))
    if data.get("model") != MODEL or integer_field(data, "n", 2) != n:
        raise VerificationError(f"Certificate model/order mismatch: {digest}")
    coefficients = data.get("h")
    certificate = check_certificate(n, coefficients)
    encoded = (str(n) + ":" + ",".join(map(str, coefficients))).encode("ascii")
    if hashlib.sha256(encoded).hexdigest() != digest:
        raise VerificationError(f"Certificate hash mismatch: {digest}")
    return certificate


def check_coverage(n: int, prefix: tuple[int, ...], witnesses):
    """Match a witness to every independently generated representative.

    Each witness supplies a checked Certificate and a complement flag.
    Complementation reverses multiplicities. A lexicographically larger
    multiplicity vector gives a smaller expanded degree sequence, so the
    chosen representatives satisfy counts >= counts[::-1].
    """
    total = representatives = 0
    previous = None
    for counts in degree_vectors(n, prefix):
        if previous is not None and counts >= previous:
            raise VerificationError("Degree enumeration is not strictly ordered")
        previous = counts
        total += 1
        complement = counts[::-1]
        if counts < complement:
            continue
        try:
            certificate, flip = next(witnesses)
        except StopIteration:
            raise VerificationError(f"Missing certificate assignment for degree sequence {expanded(counts)}") from None
        value = certificate.evaluate(complement if flip else counts)
        if value >= 0:
            raise VerificationError(f"Certificate does not exclude {expanded(counts)}: h^T b = {value}")
        representatives += 1
    if next(witnesses, None) is not None:
        raise VerificationError("Extra certificate assignments after the last degree sequence")
    return total, representatives


def verify_task(work: Path, n: int, prefix: tuple[int, ...]):
    """Verify a completed task without trusting its stored enumeration counts."""
    path = work / "parts" / task_name(prefix)
    data = require_object(load_json(path / "complete.json"), str(path / "complete.json"))
    stored_prefix = data.get("prefix")
    if (data.get("complete") is not True or not isinstance(stored_prefix, list)
            or any(type(m) is not int for m in stored_prefix) or stored_prefix != list(prefix)):
        raise VerificationError(f"Unfinished or mismatched task: {path.name}")
    expected = (integer_field(data, "degree_sequence_count"), integer_field(data, "representative_count"))
    index = load_json(path / "index.json")
    if not isinstance(index, list):
        raise VerificationError(f"Certificate index must be a list: {path}")
    for digest in index:
        require_digest(digest)
    if len(index) != len(set(index)):
        raise VerificationError(f"Duplicate certificate identifiers: {path}")
    used = {}

    with (path / "witness.bin").open("rb", buffering=1 << 20) as stream:
        def witnesses():
            # Four-byte little-endian code: 2*(certificate index + 1) + flag.
            while raw := stream.read(4):
                if len(raw) != 4:
                    raise VerificationError(f"Truncated witness in {path.name}")
                code = struct.unpack("<I", raw)[0]
                position = code // 2 - 1
                if code < 2 or position >= len(index):
                    raise VerificationError(f"Invalid certificate index in {path.name}")
                digest = index[position]
                certificate = load_certificate(work, n, digest)
                used[digest] = certificate.columns
                yield certificate, bool(code % 2)

        actual = check_coverage(n, prefix, witnesses())
    if actual != expected:
        raise VerificationError(f"Enumeration totals disagree in {path.name}: got {actual}, recorded {expected}")
    return actual[0], actual[1], used


@dataclass
class Report:
    n: int
    degree_sequences: int = 0
    representatives: int = 0
    used_certificates: int = 0
    column_checks: int = 0


def verify_directory(path: Path) -> Report:
    """Verify every required task, including tasks omitted from stored records."""
    info = require_object(load_json(path / "run.json"), "run.json")
    if info.get("format") != DIRECTORY_FORMAT or info.get("model") != MODEL:
        raise VerificationError("Unknown proof directory format or model")
    n = integer_field(info, "n", 2)
    report = Report(n)
    if info.get("analytic_obstruction") is not None:
        raise VerificationError("Directory does not contain an LP certificate proof")
    require_lp_order(n)
    depth = integer_field(info, "split_depth")
    if depth > min(3, n - 4):
        raise VerificationError("Invalid task split depth")
    prefixes = list(degree_vectors(n, length=depth))
    for prefix in prefixes:
        name = task_name(prefix)
        if not (path / "parts" / name / "complete.json").is_file():
            raise VerificationError(f"Incomplete proof: missing completed task {name}")

    load_certificate.cache_clear()
    used = {}
    for completed, prefix in enumerate(prefixes, start=1):
        total, representatives, certificates = verify_task(path, n, prefix)
        report.degree_sequences += total
        report.representatives += representatives
        used.update(certificates)
        if completed % 10 == 0 or completed == len(prefixes):
            print(f"Checked {completed}/{len(prefixes)} tasks; "
                  f"{report.representatives:,} representatives covered.", flush=True)
    report.used_certificates = len(used)
    report.column_checks = sum(used.values())
    return report


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1].startswith("-"):
        print("Usage: python snn_verify.py RUN_DIRECTORY", file=sys.stderr)
        return 2
    try:
        path = Path(sys.argv[1]).resolve()
        if not path.is_dir():
            raise VerificationError(f"Expected a proof directory: {path}")
        report = verify_directory(path)
    except KeyboardInterrupt:
        print("INTERRUPTED: verification did not complete; no conclusion is certified.", file=sys.stderr)
        return 130
    except (OSError, UnicodeError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        print(f"NOT VERIFIED: {exc}", file=sys.stderr)
        return 1

    # Each used certificate contributes its model columns once to the summary.
    print(f"n={report.n}: {report.degree_sequences:,} degree sequences; "
          f"{report.representatives:,} representatives; "
          f"{report.used_certificates:,} used integer certificates; "
          f"{report.column_checks:,} column checks.")
    print(f"VERIFIED: no simple graph has Laplacian spectrum S_{{{report.n},{report.n}}}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
