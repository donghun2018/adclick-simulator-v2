{
	"save_history": True,
	"print_simulator_output": False,
	"processing": "serial",
	"num_cores": 4,
	"master_seed": 4444,
	"date_from": "2018-01-01",
	"date_to": "2018-01-15",
	"n_experiment_reps": 1,
	"policies": [
		{
            "name": "Player",
            "policy": "PolicyThompsonSamplingSI",
            "params": {"seed": 9090}
        },
		{
            "name": "Random 1",
            "policy": "Policy2019",
            "params": {"seed": 1234}
        },
		{
            "name": "Random 2",
            "policy": "Policy2019",
            "params": {"seed": 64323}
        },
		{
            "name": "Random 3",
            "policy": "Policy2019",
            "params": {"seed": 418}
        },
		{
            "name": "Random 4",
            "policy": "Policy2019",
            "params": {"seed": 2019}
        },
		{
            "name": "Random 5",
            "policy": "Policy2019",
            "params": {"seed": 1234567}
        },
		{
            "name": "Random 6",
            "policy": "Policy2019",
            "params": {"seed": 2323}
        },
		{
            "name": "Random 7",
            "policy": "Policy2019",
            "params": {"seed": 98765}
        }
	],
	"attributes": {
		"names": ["gender", "age", "location"],
        "vals": {"gender": ["M", "F", "U"],
                 "age": ["0-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-*"],
                 "location": [
					"usa-texas-goliad", 
					"usa-california-santa_barbara",
					"usa-texas-lipscomb",
					"usa-virginia-mathews",
					"usa-texas-henderson",
					"usa-texas-delta",
					"usa-texas-martin",
					"usa-california-lake",
					"usa-virginia-clarke",
					"usa-new_jersey-sussex",
					"usa-california-san_francisco",
					"usa-california-contra_costa",
					"usa-california-san_diego",
					"usa-virginia-chesapeake_city",
					"usa-california-orange",
					"usa-virginia-manassas_city",
					"usa-new_jersey-hunterdon",
					"usa-virginia-fairfax_city",
					"usa-virginia-arlington",
					"usa-california-marin",
					"usa-texas-victoria",
					"usa-texas-burleson",
					"usa-texas-scurry",
					"usa-california-butte",
					"usa-texas-sabine",
					"usa-texas-foard",
					"usa-virginia-brunswick",
					"usa-texas-la_salle",
					"canada-nova_scotia-cumberland",
					"canada-nova_scotia-annapolis",
					"canada-nova_scotia-antigonish",
					"canada-nova_scotia-colchester",
					"canada-ontario-essex",
					"canada-ontario-hastings",
					"canada-ontario-perth",
					"canada-ontario-middlesex",
					"mexico-sonora-saric",
					"mexico-sonora-nogales",
					"mexico-sonora-altar",
					"mexico-sonora-caborca"]}
	},
	"experiment_specs": ["test"]
}