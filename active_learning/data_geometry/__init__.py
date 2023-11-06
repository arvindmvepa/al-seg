from active_learning.data_geometry.kcenter_greedy import kCenterGreedy, GPUkCenterGreedy, ProbkCenterGreedy


coreset_algs = {"kcenter_greedy": kCenterGreedy, "gpu_kcenter_greedy": GPUkCenterGreedy,
                "pkcenter_greedy": ProbkCenterGreedy}