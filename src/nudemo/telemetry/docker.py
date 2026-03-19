from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

_SIZE_RE = re.compile(r"^\s*(?P<value>[0-9.]+)\s*(?P<unit>[A-Za-z]+)?\s*$")
_DECIMAL_UNITS = {
    "B": 1,
    "kB": 1000,
    "MB": 1000**2,
    "GB": 1000**3,
    "TB": 1000**4,
}
_BINARY_UNITS = {
    "KiB": 1024,
    "MiB": 1024**2,
    "GiB": 1024**3,
    "TiB": 1024**4,
}


@dataclass(slots=True)
class ComposeService:
    service: str
    container_name: str


@dataclass(slots=True)
class ServiceSnapshot:
    snapshot_label: str
    service: str
    container_name: str
    observed_at: str
    cpu_percent: float | None
    mem_percent: float | None
    mem_usage_bytes: int | None
    mem_limit_bytes: int | None
    net_input_bytes: int | None
    net_output_bytes: int | None
    block_input_bytes: int | None
    block_output_bytes: int | None
    pids: int | None

    def to_dict(self) -> dict[str, object]:
        return {
            "snapshot_label": self.snapshot_label,
            "service": self.service,
            "container_name": self.container_name,
            "observed_at": self.observed_at,
            "cpu_percent": self.cpu_percent,
            "mem_percent": self.mem_percent,
            "mem_usage_bytes": self.mem_usage_bytes,
            "mem_limit_bytes": self.mem_limit_bytes,
            "net_input_bytes": self.net_input_bytes,
            "net_output_bytes": self.net_output_bytes,
            "block_input_bytes": self.block_input_bytes,
            "block_output_bytes": self.block_output_bytes,
            "pids": self.pids,
        }


def parse_percentage(raw: str) -> float | None:
    value = raw.strip().rstrip("%")
    return float(value) if value else None


def parse_byte_size(raw: str) -> int | None:
    match = _SIZE_RE.match(raw)
    if match is None:
        return None
    value = float(match.group("value"))
    unit = match.group("unit") or "B"
    if unit in _DECIMAL_UNITS:
        return int(value * _DECIMAL_UNITS[unit])
    if unit in _BINARY_UNITS:
        return int(value * _BINARY_UNITS[unit])
    return None


def parse_size_pair(raw: str) -> tuple[int | None, int | None]:
    left, _, right = raw.partition("/")
    return parse_byte_size(left), parse_byte_size(right)


def parse_compose_services(raw: str) -> list[ComposeService]:
    services: list[ComposeService] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        services.append(
            ComposeService(
                service=str(payload["Service"]),
                container_name=str(payload["Name"]),
            )
        )
    return services


def parse_stats_lines(
    raw: str,
    *,
    snapshot_label: str,
    service_lookup: dict[str, str],
    observed_at: datetime,
) -> list[ServiceSnapshot]:
    snapshots: list[ServiceSnapshot] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        name = str(payload["Name"])
        mem_usage_bytes, mem_limit_bytes = parse_size_pair(str(payload["MemUsage"]))
        net_input_bytes, net_output_bytes = parse_size_pair(str(payload["NetIO"]))
        block_input_bytes, block_output_bytes = parse_size_pair(str(payload["BlockIO"]))
        snapshots.append(
            ServiceSnapshot(
                snapshot_label=snapshot_label,
                service=service_lookup.get(name, name),
                container_name=name,
                observed_at=observed_at.astimezone(UTC).isoformat(),
                cpu_percent=parse_percentage(str(payload["CPUPerc"])),
                mem_percent=parse_percentage(str(payload["MemPerc"])),
                mem_usage_bytes=mem_usage_bytes,
                mem_limit_bytes=mem_limit_bytes,
                net_input_bytes=net_input_bytes,
                net_output_bytes=net_output_bytes,
                block_input_bytes=block_input_bytes,
                block_output_bytes=block_output_bytes,
                pids=int(payload["PIDs"]) if str(payload["PIDs"]).isdigit() else None,
            )
        )
    return snapshots


def capture_service_snapshots(
    compose_file: str | Path,
    *,
    snapshot_label: str,
) -> list[ServiceSnapshot]:
    compose_path = Path(compose_file)
    observed_at = datetime.now(UTC)
    ps_output = subprocess.run(
        ["docker", "compose", "-f", str(compose_path), "ps", "--format", "json"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    services = parse_compose_services(ps_output)
    if not services:
        return []

    container_names = [service.container_name for service in services]
    stats_output = subprocess.run(
        [
            "docker",
            "stats",
            "--no-stream",
            "--format",
            "{{json .}}",
            *container_names,
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    service_lookup = {
        service.container_name: service.service for service in services
    }
    return parse_stats_lines(
        stats_output,
        snapshot_label=snapshot_label,
        service_lookup=service_lookup,
        observed_at=observed_at,
    )
