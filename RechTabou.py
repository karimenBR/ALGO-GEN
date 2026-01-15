M = [[9,2,7,8],
     [6,4,3,7],
     [5,8,1,8],
     [7,6,9,4]]

def count_cost(L, M):
    return sum(M[i][L[i]] for i in range(len(L)))

def switch_items(x, y, L):
    L_copy = L[:]
    L_copy[x], L_copy[y] = L_copy[y], L_copy[x]
    return L_copy

def opt(L, M):
    B = L[:]
    C = count_cost(B, M)
    improved = True
    while improved:
        improved = False
        for i in range(len(B)-1):
            S = switch_items(i, i+1, B)
            cost_S = count_cost(S, M)
            if cost_S < C:
                B, C = S, cost_S
                improved = True
    return B, C

    
def recherche_Tabou(L,M,LV,I):
    Sbest = L[:]
    return 0

L = [0,1,2,3]
B, C = opt(L, M)
print("Permutation optimale :", B)
print("Coût minimal :", C)
