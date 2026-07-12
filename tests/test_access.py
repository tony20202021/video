"""Тесты IpAllowlist и log_ip_denied."""

from __future__ import annotations

import logging

from common.utils.access import IpAllowlist, is_ip_allowed, log_ip_denied, parse_allowed_ips


def test_parse_single_ip():
    rules = parse_allowed_ips("198.51.100.115")
    assert rules is not None
    assert rules.allows("198.51.100.115")
    assert rules.allows("127.0.0.1")
    assert not rules.allows("10.0.0.1")


def test_parse_cidr_subnet():
    rules = parse_allowed_ips("198.51.100.0/24")
    assert rules.allows("198.51.100.115")
    assert rules.allows("198.51.100.1")
    assert not rules.allows("198.51.101.1")


def test_empty_means_no_restriction():
    assert parse_allowed_ips("") is None
    assert is_ip_allowed("1.2.3.4", None)


def test_log_ip_denied_with_valid_key(caplog):
    caplog.set_level(logging.WARNING)
    log_ip_denied(
        logging.getLogger("test"),
        client_ip="198.51.100.200",
        service="Transfer",
        path="/file",
        api_key_valid=True,
    )
    assert "API key OK" in caplog.text
    assert "198.51.100.0/24" in caplog.text
