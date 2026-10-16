"""Per-instance deployment ports must reach every service and registration."""

from unittest.mock import MagicMock

import pytest
import yaml


@pytest.mark.parametrize(
    "config,env,expected",
    [
        ({}, {}, (15720, 15721)),
        ({"mcp_port": 18020, "ws_port": 18021}, {}, (18020, 18021)),
        ({"mcp_port": 15720, "ws_port": 15721},
         {"MCP_PORT": "16220", "WS_PORT": "16221"}, (16220, 16221)),
        ({"mcp_port": 18020, "ws_port": 18021},
         {"MCP_PORT": "16220"}, (16220, 18021)),
        ({"mcp_port": 18020, "ws_port": 18021},
         {"WS_PORT": "16221"}, (18020, 16221)),
    ],
)
def test_main_uses_deployment_ports(monkeypatch, tmp_path, config, env, expected):
    import main

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    monkeypatch.setenv("CONFIG_PATH", str(config_path))
    for name in ("MCP_PORT", "WS_PORT"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    monkeypatch.setattr(main.rclpy, "init", MagicMock())
    monkeypatch.setattr(main.rclpy, "shutdown", MagicMock())
    monkeypatch.setattr(main.rclpy.executors, "MultiThreadedExecutor", MagicMock())
    monkeypatch.setattr(main, "PerceptionBundle", MagicMock())
    monkeypatch.setattr(main.threading, "Thread", MagicMock())
    registration = MagicMock()
    monkeypatch.setattr(main, "_start_registration", registration)
    server = MagicMock()
    monkeypatch.setattr(main, "ThreadingHTTPServer", server)
    monkeypatch.setattr(main.signal, "signal", MagicMock())
    monkeypatch.setattr(main, "_bundle", None)

    main.main()

    mcp_port, ws_port = expected
    assert server.call_args.args[0] == ("", mcp_port)
    registration.assert_called_once_with(mcp_port, "Perception Stack", "perception")
    ws_threads = [call for call in main.threading.Thread.call_args_list
                  if call.kwargs.get("name") == "ws_asr"]
    assert len(ws_threads) == 1
    assert ws_threads[0].kwargs["args"] == (ws_port,)
