class EnvError(Exception):
    """Base exception for environment errors."""
    pass

class EnvironmentNotResetError(EnvError):
    """Raised when stepping an environment before resetting it."""
    pass

class EnvironmentDoneError(EnvError):
    """Raised when stepping an environment that has already terminated."""
    pass