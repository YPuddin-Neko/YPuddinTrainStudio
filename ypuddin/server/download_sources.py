"""Package source probes use the same outbound proxy as installation requests."""

from .network import ProxyPolicy


def probe_options(context):
    policy = ProxyPolicy.from_context(context)
    return {"opener_factory": policy.opener, "cache_key": policy}
