from Crypto.Util.number import getPrime, inverse, long_to_bytes, bytes_to_long
import random
import socket
import secrets
from secret import flag

bits = 512

def generate_random_shuffle(seed, splits):
    random.seed(seed)
    lst = list(range(bits))
    caca = [lst[j:j+splits] for j in range(0,len(lst),splits)]
    for chunk in caca:
        random.shuffle(chunk)
    return [caca for sublist in caca for caca in sublist]



def shuffle_chunk(x, shuffle_arr):
    res = 0
    for i in range(bits):
        bit = (x >> i) & 1
        res |= bit << shuffle_arr[i]
    return res




def narnia(secret, seed1, seed2, seed3, seed4, seed5):
    if not (seed1 > 0 and seed2 > 0 and seed3 > 0 and seed4 > 0 and seed5 > 0):
        return "Invalid seeds"
    if len(set([seed1, seed2, seed3, seed4, seed5])) != 5:
        return "Invalid seeds"
    xored = 0
    seed_count = 20

    random.seed(seed1)
    arr = [random.randint(0, 2**32 - 1) for _ in range(seed_count)]
    random.seed(secrets.randbits(32))
    random.shuffle(arr)
    arr = arr[:-3]
    for i in range(len(arr)):
        xored ^= arr[i]
    
    random.seed(seed2)
    arr = [random.randint(0, 2**32 - 1) for _ in range(seed_count)]
    random.seed(secrets.randbits(32))
    random.shuffle(arr)
    arr = arr[:-3]
    for i in range(len(arr)):
        xored ^= arr[i]
    
    random.seed(seed3)
    arr = [random.randint(0, 2**32 - 1) for _ in range(seed_count)]
    random.seed(secrets.randbits(32))
    random.shuffle(arr)
    arr = arr[:-3]
    for i in range(len(arr)):
        xored ^= arr[i]
    
    random.seed(seed4)
    arr = [random.randint(0, 2**32 - 1) for _ in range(seed_count)]
    random.seed(secrets.randbits(32))
    random.shuffle(arr)
    arr = arr[:-3]
    for i in range(len(arr)):
        xored ^= arr[i]
    
    random.seed(seed5)
    arr = [random.randint(0, 2**32 - 1) for _ in range(seed_count)]
    random.seed(secrets.randbits(32))
    random.shuffle(arr)
    arr = arr[:-3]
    for i in range(len(arr)):
        xored ^= arr[i]
    
    return secret ^ xored

HOST = ''    # listen on all interfaces
PORT = 1338

def handle_client(conn, addr):
    p=getPrime(bits)
    q=getPrime(bits)
    n=p*q
    e = 65537
    xd = bytes_to_long(flag)
    ct = pow(xd, e, n)
    #print(xd.bit_length())
    seedx = secrets.randbits(32)
    seedy = secrets.randbits(32)
    sefu_la_bani = [getPrime(128) for _ in range(32)]
    x = generate_random_shuffle(seedx, 13)
    y = generate_random_shuffle(seedy, 7)
    k = getPrime(256)
    cei_ce_au_valoarea = sum([i * (j+2) * (k+2) for i, j, k in zip(sefu_la_bani, x[:128][::8], y[:16])]) % k
    leak = shuffle_chunk(p, x) ^ shuffle_chunk(q, y)
    cnt = 1
    f = conn.makefile('rwb')
    try:
        while True:
            raw = f.readline()
            if not raw:
                break

            cmd = raw.strip().decode('utf-8', errors='ignore')
            if cmd == "get_data":
                print(f"[DEBUG] seedx={seedx} seedy={seedy}") # DEBUG
                f.write(b"n = " +str(n).encode() + b"\n")
                f.write(b"leak = " + str(leak).encode() + b"\n")
                f.write(b"sefu_la_bani = " + str(sefu_la_bani).encode() + b"\n")
                f.write(b"cei_ce_au_valoarea = " + str(cei_ce_au_valoarea).encode() + b"\n")
                f.write(b'k = ' + str(k).encode() + b'\n')
                f.write(b'ct = ' + str(ct).encode() + b'\n')
                f.flush()

            elif cmd.startswith("query "):
                if cnt > 42:
                    f.write(b"Query limit reached\n")
                    f.flush()
                    continue
                cnt += 1
                query_type = cmd.split()[1]
                if query_type == "x":
                    secret = seedx
                elif query_type == "y":
                    secret = seedy
                else:
                    f.write(b"Invalid query type\n")
                    f.flush()
                    continue
                seeds = cmd.split()[2:]
                if len(seeds) != 5:
                    f.write(b"Invalid number of seeds\n")
                    f.flush()
                    continue
                seeds = [int(seed) for seed in seeds]
                output = narnia(secret, *seeds)
                f.write(f"Output: {output}\n".encode())
                f.flush()


            else:
                print(f"[CMD] unknown → {cmd!r}")
                f.write(b"Unknown command\n")
                f.flush()
    finally:
        f.close()
        conn.close()

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    print(f"[*] Listening on port {PORT}…")

    while True:
        conn, addr = s.accept()
        handle_client(conn, addr)
