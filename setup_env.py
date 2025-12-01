import os
import sys
import subprocess


# --- 🎨 Terminal Colors ---
class Color:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


VENV_DIR = "venv"
MAIN_FILE = "main.py"
DEFAULT_PACKAGES = ["numpy", "sounddevice", "openai-whisper", "torch", "pyaudio"]

print(
    f"{Color.BLUE}{Color.BOLD}🚀 Arabic Lip Sync Project Setup Started...{Color.RESET}\n"
)

# Step 1: Create venv if it doesn't exist
if not os.path.exists(VENV_DIR):
    print(f"{Color.YELLOW}🔧 Creating virtual environment...{Color.RESET}")
    subprocess.run([sys.executable, "-m", "venv", VENV_DIR])
else:
    print(f"{Color.GREEN}✅ Virtual environment already exists.{Color.RESET}")

# Step 2: Locate pip inside venv
pip_path = (
    os.path.join(VENV_DIR, "Scripts", "pip.exe")
    if os.name == "nt"
    else os.path.join(VENV_DIR, "bin", "pip")
)

if not os.path.exists(pip_path):
    print(
        f"{Color.RED}❌ Could not find pip inside the virtual environment.{Color.RESET}"
    )
    sys.exit(1)

# Step 3: Upgrade pip
print(f"{Color.BLUE}⬆️  Upgrading pip...{Color.RESET}")
subprocess.run([pip_path, "install", "--upgrade", "pip"])

# Step 4: Ensure requirements.txt exists
if not os.path.exists("requirements.txt"):
    print(
        f"{Color.YELLOW}⚠️ requirements.txt not found. Creating one automatically...{Color.RESET}"
    )
    with open("requirements.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(DEFAULT_PACKAGES))
    print(
        f"{Color.GREEN}✅ Created requirements.txt with default packages.{Color.RESET}"
    )
else:
    print(f"{Color.GREEN}📄 Found existing requirements.txt.{Color.RESET}")

# Step 5: Install dependencies
print(f"{Color.BLUE}📦 Installing dependencies...{Color.RESET}")
subprocess.run([pip_path, "install", "-r", "requirements.txt"])

print(f"\n{Color.GREEN}✅ Environment setup complete!{Color.RESET}")

# Step 6: Generate or update requirements.txt (freeze current packages)
print(f"{Color.BLUE}🧾 Saving installed packages to requirements.txt...{Color.RESET}")
with open("requirements.txt", "w", encoding="utf-8") as f:
    subprocess.run([pip_path, "freeze"], stdout=f)

# Step 7: Run your main program automatically
python_path = (
    os.path.join(VENV_DIR, "Scripts", "python.exe")
    if os.name == "nt"
    else os.path.join(VENV_DIR, "bin", "python")
)

if not os.path.exists(MAIN_FILE):
    print(
        f"{Color.YELLOW}⚠️ {MAIN_FILE} not found in this directory. Skipping run step.{Color.RESET}"
    )
    sys.exit(0)

print(f"\n{Color.BLUE}{Color.BOLD}🚀 Running {MAIN_FILE}...{Color.RESET}\n")
subprocess.run([python_path, MAIN_FILE])

print(
    f"\n{Color.GREEN}🎉 All done! Your project is running inside the virtual environment.{Color.RESET}"
)
print(
    f"{Color.YELLOW}Tip: Next time, just run 'python setup_env.py' again to reinstall and launch automatically.{Color.RESET}"
)
