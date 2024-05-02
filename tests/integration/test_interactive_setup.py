import os
import subprocess
import pathlib


def test_basic():
    """
    Basic test that we can run ./interactive_setup.py and it successfully
    writes the .env file.
    """
    pathlib.Path(".env").unlink(missing_ok=True)
    # Sanitize environment so it doesn't already include any of the env vars
    # we are checking for (interactive_setup.py doesn't prompt for / write
    # to .env any required vars already present in the environment).
    env = os.environ.copy()
    for key in ["PINECONE_API_KEY", "PINECONE_API_HOST"]:
        env.pop(key, None)

    proc = subprocess.run(["python3", "interactive_setup.py"],
                   input="DUMMY_KEY\nDUMMY_HOST\n", text=True, timeout=10,
                          env=env)
    assert proc.returncode == 0
    dotenv = pathlib.Path(".env")
    assert dotenv.exists()
    assert dotenv.is_file()
    content = dotenv.read_text()
    assert "PINECONE_API_KEY=DUMMY_KEY" in content
    assert "PINECONE_API_HOST=DUMMY_HOST" in content
