"""Canonical ZarrSwarm names and compatibility with existing links, settings and state."""
import json
import os
import subprocess
import sys

import pytest

from zarrswarm import cli, parity
from zarrswarm.common import Identity, env, state_home
from zarrswarm.node import Node, parse_link


def test_dataset_links_accept_both_prefixes():
    for target in ("a" * 40, "a" * 40 + "+" + "b" * 40, "era5@" + "c" * 64):
        assert parse_link("zs://" + target) == parse_link("zt://" + target) == parse_link(target)
    with pytest.raises(ValueError, match="zs://"):
        parse_link("https://example.org/dataset")


def test_legacy_invites_are_saved_and_printed_with_the_canonical_scheme(tmp_path, capsys):
    target = "a" * 40 + "@data.example.org:7990?k=test-key"
    for i, prefix in enumerate(("zsnet://", "ztnet://", "")):
        assert cli.parse_zsnet(prefix + target) == ("a" * 40, "http://data.example.org:7990", "test-key")
        home = tmp_path / str(i)
        cli.main(["init", "--home", str(home), "--join", prefix + target])
        assert cli._config(home)["network"] == "zsnet://" + target
        output = capsys.readouterr().out
        assert "network   zsnet://" in output and "ztnet://" not in output
    with pytest.raises(ValueError, match="zsnet://"):
        cli.parse_zsnet("https://data.example.org:7990")


def test_default_home_preserves_existing_identity_and_honors_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    for key in ("ZS_HOME", "ZT_HOME"):
        monkeypatch.delenv(key, raising=False)
    assert state_home() == tmp_path / ".zs"
    original = Identity(tmp_path / ".zt")
    node = Node()
    assert node.home == tmp_path / ".zt" and node.ident.pk == original.pk
    (tmp_path / ".zs").mkdir()
    assert state_home() == tmp_path / ".zs"
    monkeypatch.setenv("ZT_HOME", str(tmp_path / "old-override"))
    assert state_home() == tmp_path / "old-override"
    monkeypatch.setenv("ZS_HOME", str(tmp_path / "new-override"))
    assert state_home() == tmp_path / "new-override"
    assert state_home(tmp_path / "explicit") == tmp_path / "explicit"


def test_init_creates_canonical_service_without_overwriting_the_legacy_unit(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    units = tmp_path / ".config/systemd/user"
    units.mkdir(parents=True)
    old = units / "zt-node.service"
    old.write_text("existing unit\n")
    home = tmp_path / "state"
    cli.main(["init", "--home", str(home), "--join", "zsnet://data.example.org:7881", "--service"])
    text = (units / "zs-node.service").read_text()
    assert f"-m zarrswarm.cli node --home {home}" in text
    assert old.read_text() == "existing unit\n"


@pytest.mark.parametrize("settings, expected, key", [
    ({}, "0.1", "configured-key"),
    ({"ZT_AUDIT": "0.2", "ZT_NETWORK_KEY": "legacy-key"}, "0.2", "legacy-key"),
    ({"ZT_AUDIT": "0.2", "ZS_AUDIT": "0.3", "ZT_NETWORK_KEY": "legacy-key", "ZS_NETWORK_KEY": "new-key"},
     "0.3", "new-key"),
    ({"ZT_NETWORK_KEY": "legacy-key", "ZS_NETWORK_KEY": ""}, "0.1", "configured-key"),
])
def test_environment_precedes_toml_tuning(tmp_path, monkeypatch, settings, expected, key):
    from zarrswarm import node
    clean = {k: v for k, v in os.environ.items() if k not in ("ZS_AUDIT", "ZT_AUDIT", "ZS_NETWORK_KEY", "ZT_NETWORK_KEY")}
    monkeypatch.setattr(os, "environ", {**clean, **settings})
    cli._write_config(tmp_path, {"tuning": {"audit_rate": 0.1}, "network_key": "configured-key"})

    class Captured(Exception):
        pass

    def capture(**kw):
        assert env("ZS_AUDIT") == expected
        assert kw["network_key"] == key
        raise Captured

    monkeypatch.setattr(node, "Node", capture)
    with pytest.raises(Captured):
        cli.main(["node", "--home", str(tmp_path)])


@pytest.mark.parametrize("prefix", ["ZS_", "ZT_"])
def test_settings_are_read_by_the_runtime_in_a_fresh_process(prefix):
    values = {"CTL": "http://127.0.0.1:19992", "AUDIT": "0.2", "VALUE_ID": "exact", "IDENTITY": "bytes",
              "CLIENT_MBPS": "12", "READAHEAD": "2", "CHUNK_OVERHEAD_MS": "25"}
    settings = {k: v for k, v in os.environ.items() if not k.startswith(("ZS_", "ZT_"))}
    settings.update({prefix + k: v for k, v in values.items()})
    code = """
import json
from zarrswarm import codec, jlps, node, scan, store
print(json.dumps([store.CTL, node.AUDIT_RATE, codec.VALUE_ID, scan.BYTE_IDENTITY,
                  jlps.CLIENT_BW, store.READAHEAD, jlps.CHUNK_OVERHEAD_S]))
"""
    result = subprocess.run([sys.executable, "-c", code], env=settings, check=True,
                            capture_output=True, text=True, timeout=30)
    assert json.loads(result.stdout) == [values["CTL"], 0.2, "exact", True, 12e6, 2, 0.025]


def test_canonical_settings_win_even_when_empty(monkeypatch):
    monkeypatch.setenv("ZT_NETWORK_KEY", "legacy-key")
    monkeypatch.setenv("ZS_NETWORK_KEY", "")
    assert env("ZS_NETWORK_KEY", "configured-key") == ""


def test_parity_recovers_data_from_both_markers():
    members = [("c0", "v0", None, b"first"), ("c1", "v1", None, b"second")]
    blob = parity.encode(members)
    assert blob.startswith(b"ZSR1")
    for marker in (b"ZSR1", b"ZTR1"):
        stored = marker + blob[4:]
        assert parity.restore_many([stored], {1: b"second"}, [0]) == {0: (b"first", "c0", "v0")}
    with pytest.raises(ValueError, match="parity header"):
        parity.header(b"junk" + blob[4:])
