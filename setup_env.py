"""Create the local environment and install only the fixed requirements."""
import os
from pathlib import Path
import subprocess
import sys
import venv


def main():
    if not (3, 11) <= sys.version_info[:2] <= (3, 13):
        raise SystemExit('Use Python 3.11, 3.12 or 3.13. This project is tested with Python 3.12.')
    root = Path(__file__).resolve().parent
    environment = root / '.venv'
    interpreter = environment / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    if not interpreter.exists():
        print('Creating .venv...', flush=True)
        venv.EnvBuilder(with_pip=True).create(environment)
    subprocess.run([str(interpreter), '-m', 'pip', 'install', '--disable-pip-version-check',
                    '--only-binary=:all:', '-r', str(root / 'requirements.txt')], check=True)
    print('\nReady. Select this interpreter in VS Code:')
    print(interpreter)
    print('Run run.py, or press F5 with the Bucha: full pipeline configuration.')


if __name__ == '__main__':
    main()
