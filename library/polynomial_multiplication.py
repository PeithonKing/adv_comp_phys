class Polynomial:
    def __init__(self, data):
        self.data = [[data[i], len(data)-i-1] for i in range(len(data))]
        self.refine()

    def __repr__(self):
        r = ""
        for coeff, deg in self.data:
            if coeff > 0:
                r += "+"
            coeff = str(coeff)
            deg = "^" + str(deg)
            if coeff == "1":
                coeff = ""
            elif coeff == "-1":
                coeff = "-"
            if deg == "^1":
                deg = ""
            elif deg == "^0":
                deg = "\b "
                if coeff == "":
                    coeff = "1"
            r += coeff + "x" + deg + " "
        if len(r):
            if r[0] == "+":
                return r[1:]
            else:
                return r
        else:
            return "0"

    def refine(self):
        data2 = []
        k = []
        for term in self.data:
            k.append(term[1])
        k = list(set(k))[::-1]
        for element in k:
            s = 0
            for term in self.data:
                if element == term[1]:
                    s += term[0]
            data2.append([s, element])
        # print(data2)
        self.data = data2

    def multiply(self, pol):
        pol1 = self.data
        pol2 = pol.data
        product = []
        for term1 in pol1:
            for term2 in pol2:
                t = [term1[0]*term2[0], term1[1]+term2[1]]
                product.append(t)
        return Polynomial(product)


class MultiPolynomial:
    def __init__(self, data):
        self.data = data
        self.refine()

    def __repr__(self): # [[1, 2, 0], [1, 1, 1], [1, 0, 2]]
        r = ""
        for term in self.data:
            coeff = term[0]
            degrees = term[1:]
            
            # working with coeff
            if coeff>1:
                coeff = "+" + str(coeff) + "*"
            elif coeff == 1:
                if sum(degrees):
                    coeff = "+"
                else:
                    coeff = "+1"
            else:
                coeff = str(coeff)
            r += coeff
            
            if coeff != "0":
                # working with degrees
                for var, degree in enumerate(degrees):
                    if degree != 0:
                        if degree != 1:
                            r += f"x{var+1}^{degree}*"
                        else:
                            r += f"x{var+1}*"
                if r[-1]=="*": r += "\b "
        if len(r):
            if r[0] == "+":
                return r[1:]
            else:
                return r
        else:
            return "0"

    def refine(self):
        A = self.data
        powers = []
        for term in A:
            powers.append(str(term[1:]))
        powers = list(set(powers))
        ans = []
        for power in powers:
            s = 0
            for term in A:
                if power == str(term[1:]):
                    s += term[0]
            ans.append([s]+eval(power))
        ans = sorted(ans, key=lambda x: sum(x[1:]), reverse=True)
        self.data = ans

    def multiply(self, pol):
        A = self.data
        B = pol.data
        ans = []
        for term1 in B:
            for term2 in A:
                t = []
                t.append(term1[0]*term2[0])
                for i in range(1, len(A[0])):
                    # print(f"\t\tterm1 = {term1}")
                    # print(f"\t\tterm2 = {term2}")
                    X = term1[i]+term2[i]
                    # print(f"\t{X}")
                    t.append(X)
                ans.append(t)
        # print(ans)
        return MultiPolynomial(ans)



if __name__ == "__main__":
    # pol1 = MultiPolynomial([[1, 1, 0, 0],
    #                         [1, 0, 1, 0]])

    # pol2 = MultiPolynomial([[1, 1, 0, 0],
    #                         [1, 0, 0, 1]])

    # pol3 = MultiPolynomial([[1, 0, 1, 0],
    #                         [1, 0, 0, 1]])

    # print("pol1 =", pol1)
    # print("pol2 =", pol2)
    # print("pol3 =", pol3)

    # result = pol1.multiply(pol2).multiply(pol3)

    # print("product =", result)
    
    print(Polynomial([1, 2, 5, 0, 6]))