from signal import signal, SIGPIPE, SIG_DFL  
from .dispatch import PtdbgDispatch
signal(SIGPIPE, SIG_DFL)


__all__ = ["PtdbgDispatch"]
