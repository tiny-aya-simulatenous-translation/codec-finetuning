"""Shim for protobuf >=5 compatibility with packages that use the removed
MessageFactory.GetPrototype method (e.g. visqol via versa).

Import this module *before* importing versa so the monkey-patch is in place
when visqol's C extension calls GetPrototype during initialisation.
"""

from google.protobuf import message_factory

if not hasattr(message_factory.MessageFactory, "GetPrototype"):

    def _get_prototype(self, descriptor):
        return message_factory.GetMessageClass(descriptor)

    message_factory.MessageFactory.GetPrototype = _get_prototype
