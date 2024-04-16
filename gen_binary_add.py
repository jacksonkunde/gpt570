import sys
import random

def generate_binary_sums(n):
    max_num = 2 ** n
    results = []

    # Generate all pairs (num_1, num_2)
    for num_1 in range(max_num):
        for num_2 in range(max_num):
            sum_ = num_1 + num_2
            # Convert to binary representation with n bits
            bin_num_1 = f"{num_1:b}".zfill(n)
            bin_num_2 = f"{num_2:b}".zfill(n)
            bin_sum = f"{sum_:b}".zfill(n + 1)  # Allow extra bit for overflow

            # Append the formatted string "num_1+num_2=sum"
            results.append(f"{bin_num_1}+{bin_num_2}={bin_sum}")
            
    return results
            
def gen_and_save(n, all_, filename):
    all_results = []
    
    if all_:
        for i in range(1, n+1):
            all_results.extend(generate_binary_sums(i))
            
    else:
        all_results.extend(generate_binary_sums(n))
        
    # shuffle so that the train and val is shuffled
    random.seed(7)
    random.shuffle(all_results)

    # Write results to a text file
    with open(filename, 'w') as file:
        for result in all_results:
            file.write(result + "\n")

    return all_results

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <n> <all> <filename>")
        sys.exit(1)
    
    try:
        n = int(sys.argv[1])
        if n < 0:
            raise ValueError("n must be a non-negative integer.")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
        
    if sys.argv[2] == 'True':
        all_ = True
    else:
        all_ = False
            
    filename = str(sys.argv[3])
    
    # Generate and print results to standard output
    results = gen_and_save(n, all_, filename)
    for result in results:
        print(result)
