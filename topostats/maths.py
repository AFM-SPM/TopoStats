"""Math helper functions"""

import math


def round_sig_figs(value: float, sig_figs: int = 5) -> float:
    """Round a value to a given number of significant figures."""

    if value != 0:
        # Calculate the exponent of the value
        exponent = math.floor(math.log10(abs(value)))

        # Calculate the number of dps to round to.
        # Note that the minus one is because the first significant
        # figure is not a decimal. Eg: 1.234 to 3sf is 1.23 (2dp).
        # The negative is due to values < 1 having a negative
        # exponent but needing a positive dp rounding value,
        # and numbers > 1 require a negative dp rounding value.
        # Eg: for 3sf 1234567 needs to be rounded to -4dp,
        # and 0.001234567 to 5dp.
        round_dp = -int(exponent - (sig_figs - 1))
        return round(value, round_dp)

    return 0
