import socket
import pytest
@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*a,**k): raise AssertionError("real_network_prohibited")
    monkeypatch.setattr(socket.socket,"connect",blocked)
    monkeypatch.setattr(socket,"create_connection",blocked)
