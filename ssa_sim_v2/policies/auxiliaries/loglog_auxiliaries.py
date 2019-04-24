# Fix paths for imports to work in unit tests ----------------

if __name__ == "__main__":
    
    from _fix_paths import fix_paths
    fix_paths()

# ------------------------------------------------------------

# Load libraries ---------------------------------------------

import numpy as np

# ------------------------------------------------------------


def find_optimal_bid(theta_0, theta_1, rpc, average_cpc_diff):
    """
    Find the optimal bid assuming the logistic log function as the bid --> click
    probability relation. The following objective function is maximized:

    .. math::
        \\frac{1}{1 + e^{-theta_0 - theta_1 \\ln(bid)}} (rpc - bid + average\_cpc\_diff)

    :param theta_0: Theta_0 parameter of the logistic curve.
    :param theta_1: Theta_1 parameter of the logistic curve.
    :param rpc: Revenue per click.
    :param average_cpc_diff: Estimated difference between the bid and the actual
        cost per click.
    :return: The optimal bid.
    :rtype: float
    """

    u = max(min(theta_0, -0.001), -1000000)
    v = max(min(theta_1, 1000000), 0.0001)
    a = min(max(rpc, 0.001), 1000000000)
    d = min(max(average_cpc_diff, 0.0), 1000000)
    
    # Define the derivative of the profit function
    
    df = lambda b: -(1.0 / (1 + np.exp(-u - v * np.log(b)))) \
        + ((a - b + d) * np.exp(-u - v * np.log(b)) * v) / (b * (1 + np.exp(-u - v * np.log(b)))**2)
        
    # Find first positive value starting from very big bids    
    
    b = 1000000000.0
    
    while df(b) < 0 and b > 0.01:
        b /= 2
        
    if b <= 0.01:
        
        return 0.01
    
    else:
        
        # Using bisection to find zero
        
        b_low = b
        b_high = b * 2
        
        eps = 0.01
        
        while b_high - b_low > eps:
            b_avg = (b_low + b_high) / 2
            
            if df(b_avg) < 0:
                b_high = b_avg
            else:
                b_low = b_avg
                
        return b_low
    

if __name__ == "__main__":
    
    #print(find_optimal_bid(0, 0, 0, 0))
    
    print(find_optimal_bid(-0.6392939760285917, -1.4251676868313812, 12.961876832844576, 1.213751269647543))
