    def largest_adj(m):
        def mul_max(largest, *entries):
            prod = np.prod(entries)
            
            if prod > largest:
                largest = prod

            return largest 

        largest = 0
        for r in range(len(m)):
            for c in range(len(m[0])):
                if r < len(m) - 3 and c < len(m[0]) - 3:
                    # Get diagonal top left
                    largest = mul_max(largest, *[m[r+i][c+i] for i in range(4)])

                    # Get diagonal top right
                    largest = mul_max(largest, m[r+3][c], m[r+1][c+2], m[r+2][c+1], m[r][c+3])

                if r < len(m) - 3:
                    # Get vertical products
                    largest = mul_max(largest, *[m[r+i][c] for i in range(4)])

                if c < len(m) - 3:
                    # Get horizontal products
                    largest = mul_max(largest, *[m[r][c+i] for i in range(4)])

        return largest
            
    # import the matrix called m
    m = np.load("grid.npy")
    print(largest_adj(m))
