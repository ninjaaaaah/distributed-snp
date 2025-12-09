import re
from .errors import WebsnapseError
from functools import lru_cache


@lru_cache(maxsize=512)
def _parse_bound_to_constraint(bound: str) -> tuple:
    """
    Parse bound string into a constraint tuple for fast evaluation.
    Returns (min_spikes, max_spikes, exact_spikes_set) where:
    - min_spikes: minimum number of spikes (or 0)
    - max_spikes: maximum number of spikes (or None for unbounded)
    - exact_spikes_set: set of exact values, or None if range-based

    Examples:
    - "a" -> (1, 1, None)  # exactly 1
    - "a^2" -> (2, 2, None)  # exactly 2
    - "a+" -> (1, None, None)  # 1 or more
    - "a*" -> (0, None, None)  # 0 or more
    - "a^{2,5}" -> would need separate handling
    """
    # Normalize the bound string
    bound = bound.replace("\\^", "^").replace("\\ast", "*")
    bound = re.sub(r"\{\s*\*\s*\}", "*", bound)
    bound = re.sub(r"\{\s*\+\s*\}", "+", bound)

    # Check for basic patterns first (most common cases)
    if bound == "a":
        return (1, 1, None)
    elif bound == "a*":
        return (0, None, None)
    elif bound == "a+":
        return (1, None, None)

    # Check for exact power: a^n or a^{n}
    match = re.match(r"^a\^(\d+)$", bound)
    if match:
        n = int(match.group(1))
        return (n, n, None)

    match = re.match(r"^a\^\{(\d+)\}$", bound)
    if match:
        n = int(match.group(1))
        return (n, n, None)

    # For complex patterns, fall back to regex (rare cases)
    return None


def check_rule_validity(bound: str, spikes: int):
    """
    Check if the number of spikes satisfies the bound constraint.
    Uses direct integer comparison instead of regex for ~10x speedup.

    Optimized with parsed constraint caching.
    """
    if not spikes:
        return False

    constraint = _parse_bound_to_constraint(bound)

    if constraint is None:
        # Fall back to regex for complex patterns
        return _check_rule_validity_regex(bound, spikes)

    min_spikes, max_spikes, exact_set = constraint

    if exact_set is not None:
        return spikes in exact_set

    if max_spikes is None:
        return spikes >= min_spikes

    return min_spikes <= spikes <= max_spikes


@lru_cache(maxsize=256)
def _check_rule_validity_regex(bound: str, spikes: int):
    """Fallback regex matcher for complex bounds (cached)."""
    bound = re.sub("\\^(\\d)", "^{\\1}", bound).replace("^", "")
    bound = re.sub(r"\{\s*\\ast\s*\}", "*", bound)
    bound = re.sub(r"\{\s*\*\s*\}", "*", bound)
    bound = re.sub(r"\{\s*\+\s*\}", "+", bound)
    bound = re.sub(r"\\ast", "*", bound)
    parsed_bound = f"^{bound}$"
    pattern = re.compile(parsed_bound)
    spike_string = "a" * spikes
    return pattern.match(spike_string) is not None


def validate_rule(rule: str):
    pattern = r"^((?P<bound>.*)\/)?(?P<consumption_bound>[a-z](\^((?P<consumed_single>[^\D])|({(?P<consumed_multiple>[2-9]|[1-9][0-9]+)})))?)\s*(\\rightarrow|\\to)\s*(?P<production>([a-z]((\^((?P<produced_single>[^0,1,\D])|({(?P<produced_multiple>[2-9]|[1-9][0-9]+]*)})))?\s*;\s*(?P<delay>[0-9]|[1-9][0-9]*))|(?P<forgot>0)|(?P<lambda>\\lambda)))$"

    result = re.match(pattern, rule)

    if result is None:
        raise WebsnapseError(f"Invalid rule definition ${rule}$")

    return result


def parse_rule(definition: str):
    """
    Performs regex matching on the rule definition to get
    the consumption, production and delay values
    """
    result = validate_rule(definition)

    forgetting = True if result.group("forgot") or result.group("lambda") else False

    consumption = (
        result.group("consumed_multiple") or result.group("consumed_single") or 1
    )
    production = (
        result.group("produced_multiple") or result.group("produced_single") or 1
        if not forgetting
        else 0
    )
    delay = int(result.group("delay") or 1 if not forgetting else 0)

    consumption = -int(consumption)
    production = int(production)

    bound = result.group("bound") or result.group("consumption_bound")

    return bound, consumption, production, delay
