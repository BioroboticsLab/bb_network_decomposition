location_cols = [
    "brood_area",
    "brood_area_open",
    "dance_floor",
    "honey_storage",
    "near_exit",
]

default_location_data_cols = [
    "bee_id",
    "age",
    "brood_area",
    "brood_area_open",
    "brood_area_combined",
    "dance_floor",
    "honey_storage",
    "near_exit",
]

default_supplementary_data_cols = [
    "r_squared",
    "day_activity",
    "phase",
    "amplitude",
    "days_left",
    "velocity",
    "velocity_day",
    "velocity_night",
]

location_labels = [
    "dance_floor",
    "honey_storage",
    "brood_area_combined",
    "near_exit",
]

supplementary_labels = default_supplementary_data_cols

default_factors = ["age", "network_age", "network_age_0+network_age_1"]
