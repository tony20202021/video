"""Ограничение доступа по IP и чтение .env."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from ipaddress import ip_address, ip_network
from pathlib import Path


def load_env_file(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        result[k.strip()] = v.strip()
    return result


@dataclass(frozen=True)
class IpAllowlist:
    """Whitelist: отдельные IP и/или подсети CIDR. localhost добавляется всегда."""

    networks: tuple
    hosts: frozenset[str]

    @classmethod
    def parse(cls, raw: str) -> IpAllowlist | None:
        text = (raw or "").strip()
        if not text:
            return None
        networks: list = []
        hosts: set[str] = {"127.0.0.1", "::1"}
        for part in text.split(","):
            token = part.strip()
            if not token:
                continue
            if "/" in token:
                networks.append(ip_network(token, strict=False))
            else:
                hosts.add(str(ip_address(token)))
        return cls(networks=tuple(networks), hosts=frozenset(hosts))

    def allows(self, client_ip: str | None) -> bool:
        if not client_ip:
            return False
        try:
            addr = ip_address(client_ip)
        except ValueError:
            return False
        if client_ip in self.hosts or str(addr) in self.hosts:
            return True
        return any(addr in net for net in self.networks)

    def summary(self) -> str:
        parts = sorted(self.hosts - {"127.0.0.1", "::1"})
        parts.extend(str(n) for n in self.networks)
        parts.extend(["127.0.0.1", "::1"])
        return ", ".join(parts)


def parse_allowed_ips(raw: str) -> IpAllowlist | None:
    return IpAllowlist.parse(raw)


def is_ip_allowed(client_ip: str | None, allowed: IpAllowlist | None) -> bool:
    if allowed is None:
        return True
    return allowed.allows(client_ip)


def log_ip_denied(
    logger: logging.Logger,
    *,
    client_ip: str | None,
    service: str,
    path: str = "",
    api_key_valid: bool = False,
) -> None:
    """Лог отказа по IP. api_key_valid=True — явный признак смены домашнего IP."""
    ip = client_ip or "?"
    where = f"{service} {path}".strip()
    if api_key_valid:
        hint = _suggest_allowlist_entry(client_ip)
        logger.warning(
            "%s: denied %s — API key OK, likely home IP changed. "
            "Update ALLOWED_IPS in .env (suggest: %s)",
            where,
            ip,
            hint,
        )
        print(
            f"[!] {where}: IP {ip} not in ALLOWED_IPS, but API key is valid.\n"
            f"    Home IP probably changed — add to .env: ALLOWED_IPS=...,{hint}",
            flush=True,
        )
    else:
        logger.warning("%s: denied %s", where, ip)


def _suggest_allowlist_entry(client_ip: str | None) -> str:
    if not client_ip:
        return "<client-ip>"
    try:
        addr = ip_address(client_ip)
    except ValueError:
        return client_ip
    if addr.version == 4:
        octets = str(addr).split(".")
        return f"{octets[0]}.{octets[1]}.{octets[2]}.0/24"
    return str(addr)
