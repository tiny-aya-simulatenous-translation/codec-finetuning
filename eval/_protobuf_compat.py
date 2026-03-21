"""Shim for protobuf >=5 compatibility with legacy ``GetPrototype`` callers.

Protobuf v5 removed ``MessageFactory.GetPrototype``, but some downstream
packages (notably ``visqol``, used by ``versa`` for VERSA evaluation)
still call it from compiled C extensions during module initialisation.

This module monkey-patches the missing method back onto
:class:`google.protobuf.message_factory.MessageFactory` so that those
legacy callers continue to work without modification.

Usage
-----
Import this module **before** importing ``versa`` (or any other package
that transitively triggers the ``visqol`` C extension)::

    import eval._protobuf_compat  # noqa: F401  -- side-effect import
    import versa  # now safe

The patch is a no-op if ``GetPrototype`` already exists (i.e. when
running with protobuf < 5).

License: MIT
"""

from google.protobuf import message_factory

# Only patch when the method is genuinely missing (protobuf >= 5).
if not hasattr(message_factory.MessageFactory, "GetPrototype"):

    def _get_prototype(self, descriptor):  # type: ignore[override]
        """Compatibility shim: delegate to the new ``GetMessageClass`` API.

        Args:
            descriptor: A protobuf ``Descriptor`` for the desired message
                type.

        Returns:
            The generated message class corresponding to *descriptor*.
        """
        return message_factory.GetMessageClass(descriptor)

    message_factory.MessageFactory.GetPrototype = _get_prototype  # type: ignore[attr-defined]
