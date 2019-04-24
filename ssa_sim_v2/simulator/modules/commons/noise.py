# Fix paths for imports to work in unit tests ----------------

if __name__ == "__main__":
    from _fix_paths import fix_paths

    fix_paths()

# ------------------------------------------------------------

# Load libraries ---------------------------------------------

from enum import Enum

# ------------------------------------------------------------


class NormalNoiseGenerator(Enum):
    additive = 'additive'
    multiplicative = 'multiplicative'

    def generate_value_with_noise(self, value, noise_level, rng):
        if self == self.additive:
            noise = rng.randn() * noise_level
        elif self == self.multiplicative:
            noise = rng.randn() * noise_level * value
        return value + noise
