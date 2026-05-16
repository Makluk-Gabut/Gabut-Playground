from scapy.all import sniff, IP, TCP, UDP, PcapWriter
from datetime import datetime
import os
import subprocess
import argparse
import sys
import signal

# ---------- Utility ----------
def is_root():
    return os.geteuid() == 0 if hasattr(os, "geteuid") else False

def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')

# ---------- Scan perangkat (optional) ----------
def scan_connected_devices():
    """
    Mencoba menjalankan arp-scan --localnet. Jika tidak tersedia atau gagal,
    mengembalikan list kosong.
    """
    print("Mendeteksi device yang terhubung di Wi-Fi (via arp-scan)...")
    try:
        result = subprocess.check_output(["arp-scan", "--localnet"], stderr=subprocess.STDOUT).decode(errors="ignore")
    except FileNotFoundError:
        print("arp-scan tidak ditemukan. Install arp-scan atau jalankan tanpa opsi scan.")
        return []
    except subprocess.CalledProcessError as e:
        # arp-scan mengembalikan non-zero code kadang; kita tetap ambil output jika ada
        result = e.output.decode(errors="ignore") if e.output else ""
        if not result:
            print("Gagal menjalankan arp-scan:", e)
            return []

    ips = set()
    for line in result.splitlines():
        line = line.strip()
        # Baris valid biasanya: "192.168.1.10    00:11:22:33:44:55    Vendor"
        parts = line.split()
        if not parts:
            continue
        # Cek apakah token pertama mirip IP
        tok = parts[0]
        tok_parts = tok.split('.')
        if len(tok_parts) == 4 and all(p.isdigit() and 0 <= int(p) <= 255 for p in tok_parts):
            ips.add(tok)
    ips = sorted(ips)
    print(f"Device terdeteksi: {ips}")
    return ips

# ---------- Packet callback & writer ----------
class Sniffer:
    def __init__(self, target_ips, pcap_path, write_append=True):
        self.target_ips = set(target_ips) if target_ips else set()
        self.pcap_path = pcap_path
        # PcapWriter: append mode bila file sudah ada
        self.writer = PcapWriter(pcap_path, append=write_append, sync=True)
        self.count = 0

    def close(self):
        try:
            self.writer.close()
        except Exception:
            pass

    def packet_callback(self, packet):
        # Hanya tangani paket IP (bisa dikembangkan)
        if IP in packet:
            src = packet[IP].src
            dst = packet[IP].dst
            proto = packet[IP].proto

            # Jika target_ips kosong => tangkap semua paket
            if not self.target_ips or src in self.target_ips or dst in self.target_ips:
                ts = datetime.now().strftime('%H:%M:%S')
                info = f"[{ts}] {src} -> {dst} | Proto: {proto}"
                if TCP in packet:
                    info += f" | TCP {packet[TCP].sport}->{packet[TCP].dport}"
                elif UDP in packet:
                    info += f" | UDP {packet[UDP].sport}->{packet[UDP].dport}"
                print(info)
                # Tulis paket langsung ke file (PcapWriter)
                try:
                    self.writer.write(packet)
                    self.count += 1
                except Exception as e:
                    print("Gagal menulis paket:", e)

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(description="Simple network sniffer for experiments (school use).")
    parser.add_argument("-i", "--iface", default="wlan0", help="Network interface (default: wlan0)")
    parser.add_argument("-o", "--output", default="sniffer_capture.pcap", help="Output pcap file")
    parser.add_argument("--no-scan", action="store_true", help="Jangan jalankan arp-scan otomatis")
    parser.add_argument("--targets", nargs="*", help="Daftar IP target (pisah dengan spasi), jika ingin filter khusus")
    parser.add_argument("--append", action="store_true", help="Append ke pcap yang sudah ada (default)")
    args = parser.parse_args()

    clear_terminal()
    print("== Sniffer Improved (eksperimen sekolah) ==")
    if not is_root():
        print("Perhatian: Skrip ini sebaiknya dijalankan dengan root (sudo). Beberapa operasi akan gagal tanpa hak root.")
        # kita tidak langsung exit; sniff akan error nanti kalau tidak cukup permission

    # Dapatkan daftar target IP
    if args.targets:
        target_ips = args.targets
        print("Menggunakan target IP dari argumen:", target_ips)
    elif not args.no_scan:
        target_ips = scan_connected_devices()
    else:
        target_ips = []

    # Inisialisasi sniffer
    sn = Sniffer(target_ips, args.output, write_append=True)

    # Signal handler agar rapi saat Ctrl+C
    def handle_sigint(signum, frame):
        print("\n\n[+] Capture dihentikan.")
        print(f"[+] Menutup writer dan menyimpan ke {args.output} ...")
        sn.close()
        print(f"[+] Total paket yang ditulis: {sn.count}")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_sigint)

    print(f"\nMulai sniffing pada interface {args.iface}. Output: {args.output}")
    if not target_ips:
        print("Menangkap semua paket (tidak ada filter IP).")
    else:
        print(f"Memfilter paket yang berhubungan dengan IP: {target_ips}")

    print("\n(Info) Tekan Ctrl+C untuk berhenti dan menyimpan hasil.\n")
    # Catatan: Jika interface Wi-Fi perlu mode monitor, user harus menyalakannya manual.
    try:
        sniff(iface=args.iface, prn=sn.packet_callback, store=0)
    except Exception as e:
        print("Error saat sniffing:", e)
    finally:
        sn.close()
        print(f"Selesai. Total paket yang ditulis: {sn.count}")

if __name__ == "__main__":
    main()
