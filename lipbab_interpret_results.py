import numpy as np
np.set_printoptions(suppress=True)

def round_lip(lip):
    if lip > 1000: return round(lip, 0)
    else: return round(lip, 2)

def print_line(arr, layers, prob):
    print(str(layers)+" layers, $\\rho = "+str(prob)+"$ ", *[round_lip(x) for x in [arr[0], arr[3], arr[6], arr[7]]], 
             *[round(arr[5], 1), int(arr[8]+0.5)], sep=' & ')
    
results_policy = []
results_certificate = []

types = 4
seeds = 5
for _ in range(types):
    policy = np.zeros(9)
    certificate = np.zeros(9)
    for _ in range(seeds):
        for _ in range(3): input()
        for i in range(2):
            for _ in range(4): input()
            inputs = input().split()
            lip = float(inputs[2][7:-1])
            time = float(inputs[-1])
            for _ in range(3): input()
            inputs = input().split()
            time_jitted = float(inputs[-1])
            input()
            res = input().split()
            first = float(res[0])
            time_first = float(res[2])
            time_lower = 600
            while len(res) == 3:
                resold = res
                if time_lower == 600 and float(res[0]) < lip:
                    time_lower = float(res[2])
                res = input().split()
            if time_lower == 600:
                print("ours better!")
            ub_final = float(resold[0])
            lb_final = float(resold[1])
            input()
            time_final = float(input())
            if i == 0: policy += np.array([lip, time, time_jitted*1000, first, time_first, time_lower, ub_final, lb_final, time_final])
            else: certificate += np.array([lip, time, time_jitted*1000, first, time_first, time_lower, ub_final, lb_final, time_final])
    results_policy.append((policy/5).tolist())
    results_certificate.append((certificate/5).tolist())
    
print_line(results_policy[0], 2, 0.99)
print_line(results_policy[2], 3, 0.99)
print_line(results_policy[1], 2, 0.99995)
print_line(results_policy[3], 3, 0.99995)

print_line(results_certificate[0], 2, 0.99)
print_line(results_certificate[2], 3, 0.99)
print_line(results_certificate[1], 2, 0.99995)
print_line(results_certificate[3], 3, 0.99995)
