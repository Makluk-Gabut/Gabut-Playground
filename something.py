import os
import sys
import winreg as reg

def add_to_startup(file_path, name):
    # Buka key registry untuk startup program user saat login
    key = reg.OpenKey(reg.HKEY_CURRENT_USER,
                      r"Software\Microsoft\Windows\CurrentVersion\Run",
                      0, reg.KEY_SET_VALUE)
    # Set nilai registry supaya file_path dijalankan saat startup
    reg.SetValueEx(key, name, 0, reg.REG_SZ, file_path)
    reg.CloseKey(key)
    print(f"Program {name} sudah ditambahkan ke startup.")

if __name__ == "__main__":
    # Path lengkap file Python yang ingin dijalankan saat startup
    file_path = os.path.abspath(sys.argv[0])
    program_name = "MyPersistentProgram"
    add_to_startup(file_path, program_name)
