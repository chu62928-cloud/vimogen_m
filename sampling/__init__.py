"""Checkout-first sampling package with runtime fallback modules."""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
