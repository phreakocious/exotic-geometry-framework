#!/usr/bin/env python3
"""
Network Protocols Investigation: Synthetic Traffic Patterns
=========================================================

Can exotic geometries distinguish between synthetic network protocol
byte streams (TCP, UDP, DNS, TLS) based on their header structures?

DIRECTIONS:
D1: Protocol Taxonomy — TCP vs UDP vs DNS vs TLS
D2: Sequential Structure — Real protocols vs Shuffled byte streams
D3: Entropy Sweep — TCP payload entropy (0.0 to 1.0)
D4: Robustness — TCP stream with increasing bit error rates
D5: Scale — UDP packet size sweep (varying header-to-payload ratio)
"""

import sys
import time
import numpy as np
import struct
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.investigation_runner import Runner

# ==============================================================
# CONFIG
# ==============================================================
SEED = 42
np.random.seed(SEED)

# ==============================================================
# HELPERS
# ==============================================================
def p16(v): return struct.pack(">H", v)
def p32(v): return struct.pack(">I", v)
def p8(v):  return struct.pack("B", v)

def make_ipv4(rng, proto, total_len, src=None, dst=None):
    """Generate a synthetic IPv4 header (20 bytes)."""
    version_ihl = 0x45
    tos = 0
    # total_len supplied
    pid = rng.integers(0, 65536)
    flags_frag = 0x4000 # Don't fragment
    ttl = 64
    checksum = 0 # Placeholder
    if src is None: src = rng.integers(0, 0xFFFFFFFF, dtype=np.uint32)
    if dst is None: dst = rng.integers(0, 0xFFFFFFFF, dtype=np.uint32)
    
    header = (
        p8(version_ihl) + p8(tos) + p16(total_len) +
        p16(pid) + p16(flags_frag) +
        p8(ttl) + p8(proto) + p16(checksum) +
        p32(src) + p32(dst)
    )
    return np.frombuffer(header, dtype=np.uint8)

def make_tcp_header(rng, payload_len):
    """Generate a synthetic TCP header (20 bytes)."""
    src_port = rng.integers(1024, 65535)
    dst_port = 80
    seq = rng.integers(0, 0xFFFFFFFF, dtype=np.uint32)
    ack = rng.integers(0, 0xFFFFFFFF, dtype=np.uint32)
    off_res_flags = 0x5018 # Header len 5, PSH+ACK
    window = 65535
    checksum = 0
    urg = 0
    
    header = (
        p16(src_port) + p16(dst_port) +
        p32(seq) + p32(ack) +
        p16(off_res_flags) + p16(window) +
        p16(checksum) + p16(urg)
    )
    return np.frombuffer(header, dtype=np.uint8)

def make_udp_header(rng, payload_len):
    """Generate a synthetic UDP header (8 bytes)."""
    src_port = rng.integers(1024, 65535)
    dst_port = 53
    length = 8 + payload_len
    checksum = 0
    
    header = (
        p16(src_port) + p16(dst_port) +
        p16(length) + p16(checksum)
    )
    return np.frombuffer(header, dtype=np.uint8)

# ==============================================================
# DATA GENERATORS
# ==============================================================

def generate_tcp_stream(rng, size, payload_entropy=1.0, error_rate=0.0):
    """
    Generates a stream of TCP packets.
    Payload entropy: 1.0 = random bytes, 0.0 = all zeros.
    """
    data = bytearray()
    while len(data) < size:
        payload_len = rng.integers(50, 100)
        
        # Payload
        if payload_entropy >= 0.99:
            payload = rng.bytes(payload_len)
        elif payload_entropy <= 0.01:
            payload = b'\x00' * payload_len
        else:
            # Mix random and zero
            mask = rng.random(payload_len) < payload_entropy
            # slow way but correct for synth
            p_arr = np.zeros(payload_len, dtype=np.uint8)
            rand_bytes = rng.integers(0, 256, payload_len, dtype=np.uint8)
            p_arr[mask] = rand_bytes[mask]
            payload = p_arr.tobytes()
            
        ipv4 = make_ipv4(rng, 6, 20 + 20 + payload_len)
        tcp = make_tcp_header(rng, payload_len)
        
        packet = ipv4.tobytes() + tcp.tobytes() + payload
        data.extend(packet)
        
    arr = np.frombuffer(data[:size], dtype=np.uint8).copy() # copy to ensure valid array
    
    if error_rate > 0:
        mask = rng.random(size) < error_rate
        noise = rng.integers(0, 256, size, dtype=np.uint8)
        arr[mask] = noise[mask]
        
    return arr

def generate_udp_stream(rng, size, payload_len_mean=60):
    """Generates a stream of UDP packets."""
    data = bytearray()
    while len(data) < size:
        plen = max(1, int(rng.normal(payload_len_mean, payload_len_mean/5)))
        payload = rng.bytes(plen)
        
        ipv4 = make_ipv4(rng, 17, 20 + 8 + plen)
        udp = make_udp_header(rng, plen)
        
        packet = ipv4.tobytes() + udp.tobytes() + payload
        data.extend(packet)
        
    return np.frombuffer(data[:size], dtype=np.uint8)

def generate_dns_stream(rng, size):
    """Generates synthetic DNS queries over UDP."""
    data = bytearray()
    domains = [b"google", b"example", b"wikipedia", b"gemini", b"openai"]
    tlds = [b"com", b"org", b"net", b"io"]
    
    while len(data) < size:
        # DNS Header (12 bytes)
        tx_id = rng.integers(0, 65536)
        flags = 0x0100 # Standard query
        qdcount = 1
        ancount = 0
        nscount = 0
        arcount = 0
        
        header = (p16(tx_id) + p16(flags) + p16(qdcount) +
                  p16(ancount) + p16(nscount) + p16(arcount))
        
        # Question: [len]label[len]label[0] [type] [class]
        domain = rng.choice(domains)
        tld = rng.choice(tlds)
        qname = p8(len(domain)) + domain + p8(len(tld)) + tld + b'\x00'
        qtype = 1 # A record
        qclass = 1 # IN
        
        question = qname + p16(qtype) + p16(qclass)
        dns_payload = header + question
        
        plen = len(dns_payload)
        ipv4 = make_ipv4(rng, 17, 20 + 8 + plen)
        udp = make_udp_header(rng, plen)
        
        packet = ipv4.tobytes() + udp.tobytes() + dns_payload
        data.extend(packet)
        
    return np.frombuffer(data[:size], dtype=np.uint8)

def generate_tls_handshake(rng, size):
    """Generates synthetic TLS ClientHello packets over TCP."""
    data = bytearray()
    while len(data) < size:
        # ClientHello Content
        legacy_ver = 0x0303 # TLS 1.2
        random_bytes = rng.bytes(32)
        sess_id_len = 32
        sess_id = rng.bytes(sess_id_len)
        cipher_suites_len = 32 # 16 suites
        cipher_suites = rng.bytes(cipher_suites_len)
        comp_methods = b'\x01\x00'
        
        handshake_body = (
            p16(legacy_ver) + random_bytes + 
            p8(sess_id_len) + sess_id +
            p16(cipher_suites_len) + cipher_suites +
            comp_methods
            # extensions omitted for brevity/simplicity
        )
        
        # Handshake Header: Type(1=ClientHello), Len(3)
        hs_header = b'\x01' + b'\x00' + p16(len(handshake_body))
        
        tls_fragment = hs_header + handshake_body
        
        # TLS Record Header: Type(22=Handshake), Ver(0x0301), Len
        rec_header = b'\x16' + b'\x03\x01' + p16(len(tls_fragment))
        
        full_tls_payload = rec_header + tls_fragment
        plen = len(full_tls_payload)
        
        ipv4 = make_ipv4(rng, 6, 20 + 20 + plen)
        tcp = make_tcp_header(rng, plen)
        
        packet = ipv4.tobytes() + tcp.tobytes() + full_tls_payload
        data.extend(packet)
        
    return np.frombuffer(data[:size], dtype=np.uint8)


# ==============================================================
# DIRECTIONS
# ==============================================================
def direction_1(runner):
    """D1: Protocol Taxonomy — TCP vs UDP vs DNS vs TLS"""
    print("\n" + "=" * 60)
    print("D1: PROTOCOL TAXONOMY")
    print("=" * 60)

    conditions = {}
    
    # TCP
    chunks = [generate_tcp_stream(rng, runner.data_size) 
              for rng in runner.trial_rngs()]
    conditions['TCP'] = runner.collect(chunks)
    
    # UDP
    chunks = [generate_udp_stream(rng, runner.data_size) 
              for rng in runner.trial_rngs()]
    conditions['UDP'] = runner.collect(chunks)
    
    # DNS
    chunks = [generate_dns_stream(rng, runner.data_size) 
              for rng in runner.trial_rngs()]
    conditions['DNS'] = runner.collect(chunks)
    
    # TLS
    chunks = [generate_tls_handshake(rng, runner.data_size) 
              for rng in runner.trial_rngs()]
    conditions['TLS'] = runner.collect(chunks)

    matrix, names, _ = runner.compare_pairwise(conditions)
    return dict(matrix=matrix, names=names)


def direction_2(runner):
    """D2: Sequential Structure — Real protocols vs Shuffled"""
    print("\n" + "=" * 60)
    print("D2: SEQUENTIAL STRUCTURE")
    print("=" * 60)

    results = {}
    
    for name, gen_fn in [('TCP', generate_tcp_stream), 
                         ('DNS', generate_dns_stream),
                         ('TLS', generate_tls_handshake)]:
        chunks = [gen_fn(rng, runner.data_size) for rng in runner.trial_rngs()]
        real = runner.collect(chunks)
        shuf = runner.collect(runner.shuffle_chunks(chunks))
        ns, _ = runner.compare(real, shuf)
        results[name] = ns
        print(f"  {name} vs shuffled = {ns:3d} sig")

    return dict(results=results)


def direction_3(runner):
    """D3: Entropy Sweep — TCP payload entropy (0.0 to 1.0)"""
    print("\n" + "=" * 60)
    print("D3: TCP PAYLOAD ENTROPY SWEEP")
    print("=" * 60)

    params = [0.0, 0.25, 0.5, 0.75, 1.0]
    baseline_chunks = [generate_tcp_stream(rng, runner.data_size, payload_entropy=0.5)
                       for rng in runner.trial_rngs()]
    baseline = runner.collect(baseline_chunks)

    results = {}
    for p in params:
        with runner.timed(f"entropy={p}"):
            chunks = [generate_tcp_stream(rng, runner.data_size, payload_entropy=p)
                      for rng in runner.trial_rngs(offset=int(p*100))]
            met = runner.collect(chunks)
            ns, _ = runner.compare(baseline, met)
            results[p] = ns

    return dict(results=results, params=params)


def direction_4(runner):
    """D4: Robustness — TCP stream with increasing bit error rates"""
    print("\n" + "=" * 60)
    print("D4: ROBUSTNESS (BIT ERROR RATE)")
    print("=" * 60)
    
    # Compare clean TCP against noisy TCP
    rates = [0.01, 0.05, 0.10, 0.15, 0.20]
    
    clean_chunks = [generate_tcp_stream(rng, runner.data_size, error_rate=0.0)
                    for rng in runner.trial_rngs()]
    clean = runner.collect(clean_chunks)
    
    results = {}
    for r in rates:
        with runner.timed(f"error_rate={r}"):
            chunks = [generate_tcp_stream(rng, runner.data_size, error_rate=r)
                      for rng in runner.trial_rngs(offset=int(r*1000))]
            met = runner.collect(chunks)
            ns, _ = runner.compare(clean, met)
            results[r] = ns
            
    return dict(results=results, rates=rates)


def direction_5(runner):
    """D5: Scale — UDP packet size sweep"""
    print("\n" + "=" * 60)
    print("D5: SCALE (UDP PAYLOAD SIZE)")
    print("=" * 60)
    
    # Vary payload size. Smaller payload = higher header frequency (more structure)
    sizes = [16, 64, 128, 256, 512]
    
    # Use 64 as baseline
    baseline_chunks = [generate_udp_stream(rng, runner.data_size, payload_len_mean=64)
                       for rng in runner.trial_rngs()]
    baseline = runner.collect(baseline_chunks)
    
    results = {}
    for s in sizes:
        if s == 64:
            results[s] = 0 # Baseline comparison to self
            continue
            
        with runner.timed(f"size={s}"):
            chunks = [generate_udp_stream(rng, runner.data_size, payload_len_mean=s)
                      for rng in runner.trial_rngs(offset=s)]
            met = runner.collect(chunks)
            ns, _ = runner.compare(baseline, met)
            results[s] = ns
            
    return dict(results=results, sizes=sizes)


# ==============================================================
# FIGURE
# ==============================================================
def make_figure(runner, d1, d2, d3, d4, d5):
    fig, axes = runner.create_figure(5, "Network Protocol Geometry")

    # D1: heatmap
    runner.plot_heatmap(axes[0], d1['matrix'], d1['names'], "D1: Taxonomy")

    # D2: bars
    names2 = list(d2['results'].keys())
    vals2 = [d2['results'][n] for n in names2]
    runner.plot_bars(axes[1], names2, vals2, "D2: vs Shuffled")

    # D3: line
    params = d3['params']
    sigs3 = [d3['results'][p] for p in params]
    runner.plot_line(axes[2], params, sigs3, "D3: Payload Entropy",
                     xlabel="Entropy")

    # D4: line
    rates = d4['rates']
    sigs4 = [d4['results'][r] for r in rates]
    runner.plot_line(axes[3], rates, sigs4, "D4: Error Rate Sensitivity",
                     xlabel="Bit Error Rate")
                     
    # D5: line
    sizes = d5['sizes']
    sigs5 = [d5['results'][s] for s in sizes]
    runner.plot_line(axes[4], sizes, sigs5, "D5: UDP Payload Size",
                     xlabel="Bytes")

    runner.save(fig, "network_protocols")


# ==============================================================
# MAIN
# ==============================================================
def main():
    t0 = time.time()
    runner = Runner("NetworkProtocols", mode="1d")

    print("=" * 60)
    print("NETWORK PROTOCOL GEOMETRY")
    print(f"size={runner.data_size}, trials={runner.n_trials}, "
          f"metrics={runner.n_metrics}")
    print("=" * 60)

    d1 = direction_1(runner)
    d2 = direction_2(runner)
    d3 = direction_3(runner)
    d4 = direction_4(runner)
    d5 = direction_5(runner)

    make_figure(runner, d1, d2, d3, d4, d5)

    elapsed = time.time() - t0
    print(f"\nTotal: {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    
    pw = d1['matrix'][np.triu_indices_from(d1['matrix'], k=1)]
    runner.print_summary({
        'D1': f"Taxonomy: {pw.min():.0f}-{pw.max():.0f} sig (mean {pw.mean():.0f})",
        'D2': [f"{n} vs shuffled = {v}" for n, v in d2['results'].items()],
        'D3': "Entropy: " + ", ".join(
            f"{p}→{d3['results'][p]}" for p in d3['params']),
        'D4': "Error rate: " + ", ".join(
            f"{r}→{d4['results'][r]}" for r in d4['rates']),
        'D5': "UDP size: " + ", ".join(
            f"{s}→{d5['results'][s]}" for s in d5['sizes']),
    })

if __name__ == "__main__":
    main()
