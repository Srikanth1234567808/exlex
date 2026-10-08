from exlex import protect, protect_local

print("=== remote host, FHE extra not installed ===")
print(protect().explain())
print()
print("=== local execution ===")
print(protect_local().explain())
