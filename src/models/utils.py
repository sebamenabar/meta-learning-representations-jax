import haiku as hk

kaiming_normal = hk.initializers.VarianceScaling(2.0, "fan_in", "truncated_normal")
